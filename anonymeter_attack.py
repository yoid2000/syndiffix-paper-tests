import anonymeter_mods
import argparse
import os
import pandas as pd
import json
import sys
import random
import numpy as np
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from collections import Counter
import pprint

pp = pprint.PrettyPrinter(indent=4)

if 'SDX_TEST_DIR' in os.environ:
    base_path = os.getenv('SDX_TEST_DIR')
else:
    base_path = os.getcwd()
if 'SDX_TEST_CODE' in os.environ:
    code_path = os.getenv('SDX_TEST_CODE')
    from syndiffix_tools.tables_manager import TablesManager
else:
    code_path = None
syn_path = os.path.join(base_path, 'synDatasets')
attack_path = os.path.join(base_path, 'anonymeter_attacks')
os.makedirs(attack_path, exist_ok=True)
# This is the total number of attacks that will be run
num_attacks = 500000
# This is the number of attacks per slurm job, and determines how many slurm jobs are created
num_attacks_per_job = 100
max_subsets = 200
debug = False

# These are the variants of the attack that exploits sub-tables
variants = {
            'syn_meter_vanilla':[],
            'syn_meter_modal':[],
            'syn_meter_modal_50':[],
            'syn_meter_modal_90':[],
            'base_meter_vanilla':[],
            'base_meter_modal':[],
            'base_meter_modal_50':[],
            'base_meter_modal_90':[],
}
# These are the thresholds we use to decide whether to use a prediction
col_comb_thresholds = {
                            'thresh_0':0,
                            'thresh_50':50,
                            'thresh_90':90,
}

def init_variants():
    for v_label in variants.keys():
        variants[v_label] = []

def convert_datetime_to_timestamp(df):
    for col in df.columns:
        if df[col].dtype == 'datetime64[ns]':
            df[col] = df[col].astype(int) / 10**9
    return df

def fit_encoders(dfs):
    # Get the string columns
    string_columns = dfs[0].select_dtypes(include=['object']).columns

    encoders = {col: LabelEncoder() for col in string_columns}

    for col in string_columns:
        # Concatenate the values from all DataFrames for this column
        values = pd.concat(df[col] for df in dfs).unique()
        # Fit the encoder on the unique values
        encoders[col].fit(values)

    return encoders

def transform_df(df, encoders):
    for col, encoder in encoders.items():
        df[col] = encoder.transform(df[col])
    return df

def transform_df_with_update(df, encoders):
    for col, encoder in encoders.items():
        if col in df.columns:
            unique_values = pd.Series(df[col].unique())
            unseen_values = unique_values[~unique_values.isin(encoder.classes_)]
            encoder.classes_ = np.concatenate([encoder.classes_, unseen_values])
            df[col] = encoder.transform(df[col])
    return df

def find_most_frequent_value(lst, fraction):
    if len(lst) == 0:
        return None

    # Count the frequency of each value in the list
    counter = Counter(lst)
    
    # Find the most common value and its count
    most_common_value, most_common_count = counter.most_common(1)[0]
    
    # Check if the most common value accounts for at least the given fraction of total entries
    if most_common_count / len(lst) > fraction:
        return most_common_value
    else:
        return None

def build_and_train_model(df, target_col, target_type):
    X = df.drop(target_col, axis=1)
    y = df[target_col]

    # If the target is categorical, encode it to integers
    if target_type == 'categorical':
        le = LabelEncoder()
        y = le.fit_transform(y)

    # Build and train the model
    if target_type == 'categorical':
        #print(f"building RandomForestClassifier with shape {X.shape}")
        try:
            model = RandomForestClassifier(random_state=42)
            #print("finished building RandomForestClassifier")
        except Exception as e:
            #print(f"A RandomForestClassifier error occurred: {e}")
            sys.exit(1)
    elif target_type == 'continuous':
        #print(f"building RandomForestRegressor with shape {X.shape}")
        try:
            model = RandomForestRegressor(random_state=42)
            #print("finished building RandomForestRegressor")
        except Exception as e:
            #print(f"A RandomForestRegressor error occurred: {e}")
            sys.exit(1)
    else:
        raise ValueError("target_type must be 'categorical' or 'continuous'")

    model.fit(X, y)
    return model

def attack_stats():
    with open(os.path.join(attack_path, 'attack_jobs.json'), 'r') as f:
        jobs = json.load(f)
    secrets = {3:{}, 6:{}, -1:{}}
    for job in jobs:
        nk = job['num_known']
        if job['secret'] not in secrets[nk]:
            secrets[nk][job['secret']] = job['num_runs']
        else:
            secrets[nk][job['secret']] += job['num_runs']
    
    for nk in [-1,3,6]:
        print(f"Number of secrets for {nk} known columns: {len(secrets[nk])}")
        print(f"Average count per secret: {sum(secrets[nk].values()) / len(secrets)}")

def make_config():
    ''' I want to generate num_attacks attacks. Each attack will be on a given secret
    column in a given table with given known columns. I will run multiple of these
    attacks per secret/table if necessary.
    '''

    num_known_columns = [-1,3,6]
    # Initialize attack_jobs
    attack_jobs = []

    # Loop over each directory name in syn_path
    for num_known in num_known_columns:
        attacks_so_far = 0
        while attacks_so_far < num_attacks:
            for dir_name in os.listdir(syn_path):
                dataset_path = os.path.join(syn_path, dir_name, 'anonymeter')
                # Check if dataset_path exists
                if not os.path.exists(dataset_path):
                    continue
                tm = TablesManager(dir_path=dataset_path)
                columns = list(tm.df_orig.columns)
                pid_cols = tm.get_pid_cols()
                if len(pid_cols) > 0:
                    # We can't really run the attack on time-series data
                    continue
                for secret in columns:
                    # We are only setup to test on categorical columns
                    if tm.orig_meta_data['column_classes'][secret] == 'continuous':
                        continue
                    aux_cols = []
                    if num_known != -1:
                        aux_cols = [col for col in columns if col not in secret]
                        if len(aux_cols) < num_known:
                            # We can't make enough aux_cols for the experiment, so just
                            # skip this secret
                            attacks_so_far += num_attacks_per_job
                            continue
                        aux_cols = random.sample(aux_cols, num_known)
                    attack_jobs.append({
                        'dir_name': dir_name,
                        'secret': secret,
                        'num_runs': num_attacks_per_job,
                        'num_known': num_known,
                        'aux_cols': aux_cols,
                    })
                    attacks_so_far += num_attacks_per_job
    # randomize the order in which the attack_jobs are run
    random.shuffle(attack_jobs)
    for index, job in enumerate(attack_jobs):
        job['index'] = index

    # Write attack_jobs into a JSON file
    with open(os.path.join(attack_path, 'attack_jobs.json'), 'w') as f:
        json.dump(attack_jobs, f, indent=4)

    exe_path = os.path.join(code_path, 'anonymeter_attack.py')
    venv_path = os.path.join(base_path, 'sdx_venv', 'bin', 'activate')
    slurm_dir = os.path.join(attack_path, 'slurm_out')
    os.makedirs(slurm_dir, exist_ok=True)
    slurm_out = os.path.join(slurm_dir, 'out.%a.out')
    num_jobs = len(attack_jobs) - 1
    # Define the slurm template
    slurm_template = f'''#!/bin/bash
#SBATCH --job-name=anonymeter_attack
#SBATCH --output={slurm_out}
#SBATCH --time=7-0
#SBATCH --mem=16G
#SBATCH --cpus-per-task=1
#SBATCH --array=0-{num_jobs}
arrayNum="${{SLURM_ARRAY_TASK_ID}}"
source {venv_path}
python {exe_path} $arrayNum
'''
    # write the slurm template to a file attack.slurm
    with open(os.path.join(attack_path, 'attack.slurm'), 'w', encoding='utf-8') as f:
        f.write(slurm_template)

def get_valid_combs(tm, secret_col, aux_cols):
    # We want the column combinations that containt secret_col and have at least
    # one other column. tm.catalog contains every subset combination that was created.
    if tm.catalog is None:
        tm.build_catalog()
    valid_combs = []
    for catalog_entry in tm.catalog:
        # check to see if every column in catalog_entry is in aux_cols
        if not all(col in aux_cols + [secret_col] for col in catalog_entry['columns']):
            continue
        if secret_col in catalog_entry['columns'] and len(catalog_entry['columns']) > 1:
            valid_combs.append(catalog_entry['columns'])
    return valid_combs

def do_inference_attacks(tm, secret_col, secret_col_type, aux_cols, regression, df_original, df_control, df_syn, num_runs):
    ''' df_original and df_control have all columns.
        df_syn has only the columns in aux_cols and secret_col.

        df_syn is the synthetic data generated from df_original.
        df_control is disjoint from df_original
    '''
    # Because I'm modeling the control and syn dataframes, and because the models
    # don't play well with string or datetime types, I'm just going to convert everthing
    df_original = convert_datetime_to_timestamp(df_original)
    df_control = convert_datetime_to_timestamp(df_control)
    df_syn = convert_datetime_to_timestamp(df_syn)
    encoders = fit_encoders([df_original, df_control, df_syn])

    df_original = transform_df(df_original, encoders)
    df_control = transform_df(df_control, encoders)
    df_syn = transform_df(df_syn, encoders)
    attack_cols = aux_cols + [secret_col]
    # model_base is the baseline built from an ML model
    print("build baseline model")
    model_base = build_and_train_model(df_control[attack_cols], secret_col, secret_col_type)
    # model_attack is used to generate a groundhog day type attack
    print("build attack model")
    model_attack = build_and_train_model(df_syn[attack_cols], secret_col, secret_col_type)
    # model_original is used simply to demonstrate the ineffectiveness of the groundhog attack
    print("build original model")
    model_original = build_and_train_model(df_original[attack_cols], secret_col, secret_col_type)

    num_model_base_correct = 0
    num_model_attack_correct = 0
    num_model_original_correct = 0
    num_syn_correct = 0
    num_meter_base_correct = 0
    attacks = []
    modal_value = df_original[secret_col].mode().iloc[0]
    num_modal_rows = df_original[df_original[secret_col] == modal_value].shape[0]
    modal_percentage = round(100*(num_modal_rows / len(df_original)), 2)
    print(f"start {num_runs} runs")
    for i in range(num_runs):
        init_variants()
        print(".", end='', flush=True)
        # There is a chance of replicas here, but small enough that we ignore it
        targets = df_original[attack_cols].sample(1)
        # Get the value of the secret column in the first row of targets
        secret_value = targets[secret_col].iloc[0]
        # Count the number of rows that contian secret_value in column secret_col
        num_secret_rows = df_original[secret_col].value_counts().get(secret_value, 0)
        secret_percentage = round(100*(num_secret_rows / len(df_original)), 2)
        this_attack = {
            'secret_value': str(secret_value),
            'secret_percentage': secret_percentage,
            'secret_col_type': secret_col_type,
            'modal_value': str(modal_value),
            'modal_percentage': modal_percentage,
            'num_known_cols': len(aux_cols),
            'known_cols': str(aux_cols),
        }
        # Now get the model baseline prediction
        try:
            model_base_pred_value = model_base.predict(targets.drop(secret_col, axis=1))
            model_base_pred_value = model_base_pred_value[0]
        except Exception as e:
            print(f"A model.predict() Error occurred: {e}")
            sys.exit(1)
        # convert model_base_pred_value to a series
        model_base_pred_value_series = pd.Series(model_base_pred_value, index=targets.index)
        model_base_answer = anonymeter_mods.evaluate_inference_guesses(guesses=model_base_pred_value_series, secrets=targets[secret_col], regression=regression).sum()
        if model_base_answer not in [0,1]:
            print(f"Error: unexpected answer {model_base_answer}")
            sys.exit(1)
        num_model_base_correct += model_base_answer
        this_attack['model_base_pred_value'] = str(model_base_pred_value)
        this_attack['model_base_answer'] = int(model_base_answer)

        # Now run the model attack
        try:
            model_attack_pred_value = model_attack.predict(targets.drop(secret_col, axis=1))
            model_attack_pred_value = model_attack_pred_value[0]
        except Exception as e:
            print(f"A model.predict() Error occurred: {e}")
            sys.exit(1)
        # convert model_attack_pred_value to a series
        model_attack_pred_value_series = pd.Series(model_attack_pred_value, index=targets.index)
        model_attack_answer = anonymeter_mods.evaluate_inference_guesses(guesses=model_attack_pred_value_series, secrets=targets[secret_col], regression=regression).sum()
        if model_attack_answer not in [0,1]:
            print(f"Error: unexpected answer {model_attack_answer}")
            sys.exit(1)
        num_model_attack_correct += model_attack_answer
        this_attack['model_attack_pred_value'] = str(model_attack_pred_value)
        this_attack['model_attack_answer'] = int(model_attack_answer)

        # Now run the model attack using the groundhog model
        try:
            model_original_pred_value = model_original.predict(targets.drop(secret_col, axis=1))
            model_original_pred_value = model_original_pred_value[0]
        except Exception as e:
            print(f"A model.predict() Error occurred: {e}")
            sys.exit(1)
        # convert model_original_pred_value to a series
        model_original_pred_value_series = pd.Series(model_original_pred_value, index=targets.index)
        model_original_answer = anonymeter_mods.evaluate_inference_guesses(guesses=model_original_pred_value_series, secrets=targets[secret_col], regression=regression).sum()
        if model_original_answer not in [0,1]:
            print(f"Error: unexpected answer {model_original_answer}")
            sys.exit(1)
        num_model_original_correct += model_original_answer
        this_attack['model_original_pred_value'] = str(model_original_pred_value)
        this_attack['model_original_answer'] = int(model_original_answer)

        # Run the anonymeter-style attack on the synthetic data
        syn_meter_pred_values = []
        ans = anonymeter_mods.run_anonymeter_attack(
                                        targets=targets,
                                        basis=df_syn[attack_cols],
                                        aux_cols=aux_cols,
                                        secret=secret_col,
                                        regression=regression)
        syn_meter_pred_value_series = ans['guess_series']
        syn_meter_pred_value = syn_meter_pred_value_series.iloc[0]
        syn_meter_pred_values.append(syn_meter_pred_value)
        syn_meter_answer = anonymeter_mods.evaluate_inference_guesses(guesses=syn_meter_pred_value_series, secrets=targets[secret_col], regression=regression).sum()
        if syn_meter_answer not in [0,1]:
            print(f"Error: unexpected answer {syn_meter_answer}")
            sys.exit(1)
        num_syn_correct += syn_meter_answer
        this_attack['syn_meter_pred_value'] = str(syn_meter_pred_value)
        this_attack['syn_meter_answer'] = int(syn_meter_answer)

        # Run the anonymeter-style attack on the control data for the baseline
        ans = anonymeter_mods.run_anonymeter_attack(
                                        targets=targets,
                                        basis=df_control[attack_cols],
                                        aux_cols=aux_cols,
                                        secret=secret_col,
                                        regression=regression)
        base_meter_pred_value_series = ans['guess_series']
        base_meter_pred_value = base_meter_pred_value_series.iloc[0]
        base_meter_answer = anonymeter_mods.evaluate_inference_guesses(guesses=base_meter_pred_value_series, secrets=targets[secret_col], regression=regression).sum()
        if base_meter_answer not in [0,1]:
            print(f"Error: unexpected answer {base_meter_answer}")
            sys.exit(1)
        num_meter_base_correct += base_meter_answer
        this_attack['base_meter_pred_value'] = str(base_meter_pred_value)
        this_attack['base_meter_answer'] = int(base_meter_answer)

        # Now, we want to run the anonymeter-style attack on every valid
        # synthetic dataset. We will use this additional information to decide
        # if the anonymeter-style attack on the full dataset is correct or not.
        num_subset_combs = 0
        num_subset_correct = 0
        col_combs = get_valid_combs(tm, secret_col, aux_cols)
        #print(f"Running with total {max_subsets} of {len(col_combs)} column combinations")
        if len(col_combs) > max_subsets:
            col_combs = random.sample(col_combs, max_subsets)
        this_attack['num_subsets'] = len(col_combs)
        for col_comb in col_combs:
            # First run attack on synthetic data
            df_syn_subset = tm.get_syn_df(col_comb)
            df_syn_subset = convert_datetime_to_timestamp(df_syn_subset)
            df_syn_subset = transform_df_with_update(df_syn_subset, encoders)
            subset_aux_cols = col_comb.copy()
            subset_aux_cols.remove(secret_col)
            ans_syn = anonymeter_mods.run_anonymeter_attack(
                                            targets=targets[col_comb],
                                            basis=df_syn_subset[col_comb],
                                            aux_cols=subset_aux_cols,
                                            secret=secret_col,
                                            regression=regression)
            # Compute an answer based on the vanilla anonymeter attack
            pred_value_series = ans_syn['guess_series']
            pred_value = pred_value_series.iloc[0]
            variants['syn_meter_vanilla'].append(pred_value)

            # Compute an answer based on the modal anonymeter attack
            variants['syn_meter_modal'].append(ans_syn['match_modal_value'])	

            # Compute an answer only if the modal value is more than 50% of the possible answers
            if ans_syn['match_modal_percentage'] > 50:
                variants['syn_meter_modal_50'].append(ans_syn['match_modal_value'])

            # Compute an answer only if the modal value is more than 90% of the possible answers
            if ans_syn['match_modal_percentage'] > 90:
                variants['syn_meter_modal_90'].append(ans_syn['match_modal_value'])

            # Then run attack on control data for the baseline
            ans_base = anonymeter_mods.run_anonymeter_attack(
                                            targets=targets[col_comb],
                                            basis=df_control[col_comb],
                                            aux_cols=subset_aux_cols,
                                            secret=secret_col,
                                            regression=regression)
            # Compute an answer based on the vanilla anonymeter attack
            pred_value_series = ans_base['guess_series']
            pred_value = pred_value_series.iloc[0]
            variants['base_meter_vanilla'].append(pred_value)

            # Compute an answer based on the modal anonymeter attack
            variants['base_meter_modal'].append(ans_base['match_modal_value'])	

            # Compute an answer only if the modal value is more than 50% of the possible answers
            if ans_base['match_modal_percentage'] > 50:
                variants['base_meter_modal_50'].append(ans_base['match_modal_value'])

            # Compute an answer only if the modal value is more than 90% of the possible answers
            if ans_base['match_modal_percentage'] > 90:
                variants['base_meter_modal_90'].append(ans_base['match_modal_value'])

        if debug:
            print(f"variants:")
            pp.pprint(variants)
        # We want to filter again according to the amount of agreement among the
        # different column combinations
        for v_label, pred_values in variants.items():
            if debug:
                print(f"v_label: {v_label}")
                print(f"pred_values: {pred_values}")
            for cc_label, cc_thresh in col_comb_thresholds.items():
                label = f"{v_label}_{cc_label}"
                pred_value = find_most_frequent_value(pred_values, cc_thresh/100)
                if pred_value is not None:
                    pred_value_series = pd.Series(pred_value, index=targets.index)
                    answer = anonymeter_mods.evaluate_inference_guesses(guesses=pred_value_series, secrets=targets[secret_col], regression=regression).sum()
                else:
                    answer = -1     # no prediction
                this_attack[f'{label}_value'] = str(pred_value)
                this_attack[f'{label}_answer'] = int(answer)

        if debug:
            print(f"this_attack:")
            pp.pprint(this_attack)
        attacks.append(this_attack)
        #print('---------------------------------------------------')
        #pp.pprint(attacks[-1])
    print(f"\nnum_model_base_correct: {num_model_base_correct}\nnum_syn_correct: {num_syn_correct}\nnum_meter_base_correct: {num_meter_base_correct}\nnum_model_attack_correct: {num_model_attack_correct}\nnum_model_original_correct: {num_model_original_correct}")
    return attacks


def run_attack(job_num):
    with open(os.path.join(attack_path, 'attack_jobs.json'), 'r') as f:
        jobs = json.load(f)

    # Make sure job_num is within the range of jobs, and if not, print an error message and exit
    if job_num < 0 or job_num >= len(jobs):
        print(f"Invalid job number: {job_num}")
        return

    # Get the job
    job = jobs[job_num]

    # Create 'instances' directory in attack_path if it isn't already there
    instances_path = os.path.join(attack_path, 'instances')
    os.makedirs(instances_path, exist_ok=True)

    # Make a file_name and file_path
    # make a string that contains the column names in job['columns'] separated by '_'
    file_name = f"{job['dir_name']}.{job['secret']}.{job_num}.json"
    file_path = os.path.join(instances_path, file_name)

    if os.path.exists(file_path):
        if job['num_known'] == -1:
            print(f"File already exists: {file_path}")
            return
    dataset_path = os.path.join(syn_path, job['dir_name'], 'anonymeter')
    control_path = os.path.join(dataset_path, 'control.parquet')
    # read the control file into a DataFrame
    df_control = pd.read_parquet(control_path)
    # Make a TablesManager object
    tm = TablesManager(dir_path=dataset_path)
    # First, run the attack on the full synthetic dataset
    df_syn = tm.get_syn_df()
    print(f"df_syn has shape {df_syn.shape} and columns {df_syn.columns}")
    # set aux_cols to all columns except the secret column
    aux_cols = [col for col in df_syn.columns if col not in [job['secret']]]
    if job['num_known'] != -1:
        # select num_known columns from aux_cols
        aux_cols = random.sample(aux_cols, job['num_known'])
    if tm.orig_meta_data['column_classes'][job['secret']] == 'continuous':
        regression = True
        target_type = 'continuous'
        print(f"We are no longer doing continuous secret column attacks: {job}")
        sys.exit(1)
    else:
        regression = False
        target_type = 'categorical'
    attacks = do_inference_attacks(tm, job['secret'], target_type, aux_cols, regression, tm.df_orig, df_control, df_syn, job['num_runs'])
    with open(file_path, 'w') as f:
        json.dump(attacks, f, indent=4)

def gather(instances_path):
    attacks = []
    # check to see if attacks.parquet exists
    if os.path.exists(os.path.join(attack_path, 'attacks.parquet')):
        # read it as a DataFrame
        print("Reading attacks.parquet")
        df = pd.read_parquet(os.path.join(attack_path, 'attacks.parquet'))
    else:
        all_files = list(os.listdir(instances_path))
        # loop through the index and filename of all_files
        num_files_with_num_known = {3:0, 6:0, -1:0}
        for i, filename in enumerate(all_files):
            if not filename.endswith('.json'):
                print(f"!!!!!!!!!!!!!!! bad filename: {filename}!!!!!!!!!!!!!!!!!!!!!!!")
                continue
            # split the filename on '.'
            table = filename.split('.')[0]
            with open(os.path.join(instances_path, filename), 'r') as f:
                if i % 100 == 0:
                    print(f"Reading {i+1} of {len(all_files)} {filename}")
                res = json.load(f)
                if 'num_known_cols' not in res[0]:
                    print("old format")
                    pp.pprint(res[0])
                    print(filename)
                    continue
                for record in res:
                    record['dataset'] = table
                attacks += res
                if 'num_known_cols' not in res[0]:
                    pp.pprint(res[0])
                    print(filename)
                if res[0]['num_known_cols'] == 3:
                    num_files_with_num_known[3] += len(res)
                elif res[0]['num_known_cols'] == 6:
                    num_files_with_num_known[6] += len(res)
                else:
                    num_files_with_num_known[-1] += len(res)
        print(f"Total attacks: {len(attacks)}")
        pp.pprint(num_files_with_num_known)
        # convert attacks to a DataFrame
        df = pd.DataFrame(attacks)
        for col in df.columns:
            if df[col].dtype == object:
                try:
                    df[col] = df[col].astype(int)
                except ValueError:
                    try:
                        df[col] = df[col].astype(float)
                    except ValueError:
                        pass
        # print the dtypes of df
        pp.pprint(df.dtypes)
        # save the dataframe to a parquet file
        df.to_parquet(os.path.join(attack_path, 'attacks.parquet'))
        # save the dataframe to a csv file
        df.to_csv(os.path.join(attack_path, 'attacks.csv'))
    return df

def update_max_improve(max_improve, max_info, label, stats):
    improve_label = f"{label}_improve"
    cov_label = f"{label}_coverage"
    if stats[improve_label] > max_improve:
        max_improve = stats[improve_label]
        if cov_label in stats:
            cov = stats[cov_label]
        else:
            cov = 1.0
        max_info = {'label':label, 'improve':max_improve, 'coverage':cov}
    return max_improve, max_info

def get_basic_stats(stats, df):
    max_improve = -1000
    max_info = {}
    stats['num_attacks'] = len(df)
    stats['avg_num_subsets'] = round(df['num_subsets'].mean(), 2)
    stats['average_percentage'] = round(df['secret_percentage'].mean(), 2)
    p_model = round(df['model_base_answer'].sum() / len(df), 6)
    stats['model_base_precision'] = p_model
    p_meter = round(df['base_meter_answer'].sum() / len(df), 6)
    stats['meter_base_precision'] = p_meter
    p_base = max(p_model, p_meter)
    p = round(df['model_attack_answer'].sum() / len(df), 6)
    stats['model_attack_precision'] = p
    stats['model_attack_improve'] = round((p-p_base)/(1.0000001-p_base), 6)
    p = round(df['model_original_answer'].sum() / len(df), 6)
    stats['model_original_precision'] = p
    stats['model_original_improve'] = round((p-p_base)/(1.0000001-p_base), 6)
    p = round(df['syn_meter_answer'].sum() / len(df), 6)
    stats['meter_attack_precision'] = p
    stats['meter_attack_improve'] = round((p-p_base)/(1.0000001-p_base), 6)
    max_improve, max_info = update_max_improve(max_improve, max_info, 'meter_attack', stats)
    for v_label in variants.keys():
        for cc_label in col_comb_thresholds.keys():
            base_label = f"syn_meter_{v_label}_{cc_label}"
            answer = f"{base_label}_answer"
            precision = f"{base_label}_precision"
            improve = f"{base_label}_improve"
            coverage = f"{base_label}_coverage"
            model_base = f"model_base_{v_label}_{cc_label}_precision"
            meter_base = f"meter_base_{v_label}_{cc_label}_precision"
            # df_pred contains only the rows where predictions were made
            df_pred = df[df[answer] != -1]
            if len(df_pred) == 0:
                stats[model_base] = 0
                stats[meter_base] = 0
                stats[coverage] = 0
                stats[precision] = 0
                stats[improve] = 0
                continue
            # Basing the base precision on the rows where the attack happened to make predictions
            # is not necessarily the right thing to do. What we really want is to find the best base
            # precision given a similar coverage
            p_model_pred = round(df_pred['model_base_answer'].sum() / len(df_pred), 6)
            stats[model_base] = p_model_pred
            p_meter_pred = round(df_pred['base_meter_answer'].sum() / len(df_pred), 6)
            stats[meter_base] = p_meter_pred
            p_base_pred = max(p_model_pred, p_meter_pred)
            stats[coverage] = len(df_pred) / len(df)
            p = df_pred[answer].sum() / len(df_pred)
            stats[precision] = round(p, 6)
            stats[improve] = round((p-p_base_pred)/(1.0000001-p_base_pred), 6)
            max_improve, max_info = update_max_improve(max_improve, max_info, base_label, stats)
    stats['max_improve_record'] = max_info
    stats['max_improve'] = max_info['improve']
    stats['max_coverage'] = max_info['coverage']	

def get_by_metric_from_by_slice(stats):
    problem_cases = []
    for metric in stats['by_slice']['all_results'].keys():
        stats['by_metric'][metric] = {}	
        for slice_key, result in stats['by_slice'].items():
            stats['by_metric'][metric][slice_key] = result[metric]
            if metric[-7:] == 'improve' and metric not in ['model_original_improve', 'max_improve']:
                if result[metric] > 0.5:
                    problem_cases.append(str((slice_key, metric, result[metric])))
                    cov_metric = metric[:-7] + 'coverage'
                    coverage = stats['by_metric'][cov_metric][slice_key]
                    problem_cases.append(str((slice_key, cov_metric, coverage)))
                    prec_metric = metric[:-7] + 'precision'
                    precision = stats['by_metric'][prec_metric][slice_key]
                    problem_cases.append(str((slice_key, prec_metric, precision)))
                    meter_base_metric = 'meter_base' + prec_metric[9:]
                    meter_base_precision = stats['by_metric'][meter_base_metric][slice_key]
                    model_base_metric = 'model_base' + prec_metric[9:]
                    model_base_precision = stats['by_metric'][model_base_metric][slice_key]
                    base_precision = max(meter_base_precision, model_base_precision)
                    num_predictions = int(round(coverage * stats['by_metric']['num_attacks'][slice_key]))
                    problem_cases.append(str(('base_precision', base_precision, 'num_predictions', num_predictions)))
    stats['problem_cases'] = problem_cases

def digin(df):
    df = df.copy()
    print("---------------------------------------------------")
    for low, high in [[0,10], [10,20], [20,30], [30,40], [40,50], [50,60], [60,70], [70,80], [80,90], [90,100]]:
        num_rows_high_true = df[(df['modal_value'] == df['secret_value']) & (df['modal_percentage'] > low) & (df['modal_percentage'] < high) & (df['high_syn_meter_answer'] == 1)].shape[0]
        num_rows_high_false = df[(df['modal_value'] == df['secret_value']) & (df['modal_percentage'] > low) & (df['modal_percentage'] < high) & (df['high_syn_meter_answer'] == 0)].shape[0]
        frac_true = round(100*(num_rows_high_true / (num_rows_high_true + num_rows_high_false + 0.00001)), 2)
        print(f"{low}-{high} percent true = {frac_true} ({num_rows_high_true}, {num_rows_high_false})")
    print("---------------------------------------------------")
    for low, high in [[0,10], [10,20], [20,30], [30,40], [40,50], [50,60], [60,70], [70,80], [80,90], [90,100]]:
        num_rows_true = df[(df['modal_value'] == df['secret_value']) & (df['modal_percentage'] > low) & (df['modal_percentage'] <= high)].shape[0]
        num_rows = df[(df['modal_percentage'] > low) & (df['modal_percentage'] <= high)].shape[0]
        frac_true = round(100*(num_rows_true / (num_rows + 0.00001)), 2)
        print(f"{low}-{high} precision = {frac_true} ({num_rows_true}, {num_rows})")


def df_compare(df1: pd.DataFrame, df2: pd.DataFrame) -> int:
    # Ensure the dataframes have the same columns and index
    df1, df2 = df1.align(df2)

    # Create a boolean mask where each row is True if the row in df1 differs from the row in df2
    mask = ~(df1 == df2).all(axis=1)

    # Count the number of True values in the mask
    num_differing_rows = mask.sum()

    return num_differing_rows

def set_model_base_predictions(df, thresh):
    df_copy = df.copy()
    num_pred_value_changes = 0
    num_answer_changes_to_1 = 0
    num_answer_changes_to_0 = 0
    for index, row in df_copy.iterrows():
        if row['modal_percentage'] > thresh:
            if row['model_base_pred_value'] != row['modal_value']:
                num_pred_value_changes += 1
                #print(row['model_base_pred_value'], row['modal_value'])
                df_copy.at[index, 'model_base_pred_value'] = row['modal_value']
            if row['modal_value'] == row['secret_value']:
                if row['model_base_answer'] != 1:
                    num_answer_changes_to_1 += 1
                df_copy.at[index, 'model_base_answer'] = 1
            else:
                if row['model_base_answer'] != 0:
                    num_answer_changes_to_0 += 1
                df_copy.at[index, 'model_base_answer'] = 0
    return df_copy

def run_stats_for_subsets(stats, df, num_subsets):
    stats['by_slice']['all_results'] = {}
    get_basic_stats(stats['by_slice']['all_results'], df)
    # make a new df that contains only rows where 'secret_col_type' is 'categorical'
    df_cat = df[df['secret_col_type'] == 'categorical']
    df_cat_copy = df_cat.copy()
    stats['by_slice']['categorical_results'] = {}
    get_basic_stats(stats['by_slice']['categorical_results'], df_cat_copy)
    #df_cat_copy['percentile_bin'] = pd.qcut(df_cat_copy['secret_percentage'], q=10, labels=False)
    df_cat_copy['percentile_bin'] = pd.cut(df_cat_copy['modal_percentage'], bins=10, labels=False)
    for bin_value, df_bin in df_cat_copy.groupby('percentile_bin'):
        average_percentage = round(df_bin['secret_percentage'].mean(), 2)
        slice_name = f"cat_modal_percentage_{average_percentage}"
        stats['by_slice'][slice_name] = {}
        get_basic_stats(stats['by_slice'][slice_name], df_bin)
    for bin_value, df_bin in df_cat_copy.groupby('dataset'):
        slice_name = f"cat_dataset_{bin_value}"
        stats['by_slice'][slice_name] = {}
        get_basic_stats(stats['by_slice'][slice_name], df_bin)
    if False:
        df_70 = set_model_base_predictions(df_cat, 70)
        stats['by_slice']['categorical_results_70'] = {}
        get_basic_stats(stats['by_slice']['categorical_results_70'], df_70)
        df_70['percentile_bin'] = pd.cut(df_70['modal_percentage'], bins=10, labels=False)
        for bin_value, df_bin in df_70.groupby('percentile_bin'):
            average_percentage = round(df_bin['secret_percentage'].mean(), 2)
            slice_name = f"cat_70_modal_percentage_{average_percentage}"
            stats['by_slice'][slice_name] = {}
            get_basic_stats(stats['by_slice'][slice_name], df_bin)
        for bin_value, df_bin in df_70.groupby('dataset'):
            slice_name = f"cat_70_dataset_{bin_value}"
            stats['by_slice'][slice_name] = {}
            get_basic_stats(stats['by_slice'][slice_name], df_bin)
    #digin(df_cat)
    get_by_metric_from_by_slice(stats)
    #pp.pprint(stats)

def do_plots():
    df = gather(instances_path=os.path.join(attack_path, 'instances'))

    print(f"df has shape {df.shape} and columns:")
    print(df.head())
    for col in df.columns:
        if df[col].dtype == object:
            try:
                df[col] = df[col].astype(int)
            except ValueError:
                try:
                    df[col] = df[col].astype(float)
                except ValueError:
                    pass
    # print the columns and dtypes of df
    print(df.dtypes)
    # print the distinct values in modal_value, secret_value, and model_base_pred_value
    stats = {}
    # print the count of each distinct value in num_known_cols
    print(df['num_known_cols'].value_counts())
    for sub_key, num_known in [('num_known_all', -1), ('num_known_3', 3), ('num_known_6', 6)]:
        if num_known != -1:
            df_copy = df[df['num_known_cols'] == num_known].copy()
        else:
            df_copy = df[(df['num_known_cols'] != 3) & (df['num_known_cols'] != 6)].copy()
        stats[sub_key] = {'by_slice': {}, 'by_metric': {}}
        run_stats_for_subsets(stats[sub_key], df_copy, num_known)
    # save stats as json file
    print(f"Writing stats to {os.path.join(attack_path, 'stats.json')}")
    with open(os.path.join(attack_path, 'stats.json'), 'w') as f:
        json.dump(stats, f, indent=4)

def do_tests():
    if find_most_frequent_value([1, 2, 2, 3, 3, 3], 0.5) != 3:
        print("failed 1")
    if find_most_frequent_value([1, 2, 2, 3, 3, 3], 0.6) is not None:
        print("failed 2")
    if find_most_frequent_value([], 0.5) is not None:
        print("failed 3")
    if find_most_frequent_value([1, 1, 1, 1, 1], 0.2) != 1:
        print("failed 4")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", help="'config' to run make_config(), or an integer to run run_attacks()")
    args = parser.parse_args()

    if args.command == 'config':
        make_config()
    if args.command == 'stats':
        attack_stats()
    if args.command == 'test':
        do_tests()
    elif args.command == 'plots':
        do_plots()
    else:
        try:
            job_num = int(args.command)
            run_attack(job_num)
        except ValueError:
            print(f"Unknown command: {args.command}")

if __name__ == "__main__":
    main()