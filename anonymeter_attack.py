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
num_attacks = 100000
num_attacks_per_job = 50
max_subsets = 200

from sklearn.preprocessing import LabelEncoder

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
    if most_common_count / len(lst) >= fraction:
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

def make_config():
    ''' I want to generate num_attacks attacks. Each attack will be on a given secret
    column in a given table. I will run multiple of these attacks per secret/table if
    necessary.
    '''
    # Initialize attack_jobs
    attack_jobs = []

    # Loop over each directory name in syn_path
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
                attack_jobs.append({
                    'dir_name': dir_name,
                    'secret': secret,
                    'num_runs': num_attacks_per_job,
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

def get_valid_combs(tm, secret_col):
    # We want the column combinations that containt secret_col and have at least
    # one other column
    if tm.catalog is None:
        tm.build_catalog()
    valid_combs = []
    for catalog_entry in tm.catalog:
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

    num_model_base_correct = 0
    num_model_attack_correct = 0
    num_syn_correct = 0
    num_meter_base_correct = 0
    attacks = []
    for i in range(num_runs):
        # There is a chance of replicas here, but small enough that we ignore it
        targets = df_original[attack_cols].sample(1)
        # Get the value of the secret column in the first row of targets
        secret_value = targets[secret_col].iloc[0]
        # Count the number of rows that contian secret_value in column secret_col
        num_secret_rows = df_original[secret_col].value_counts().get(secret_value, 0)
        secret_percentage = round(100*(num_secret_rows / len(df_original)), 2)

        # Now get the model baseline prediction
        try:
            model_base_pred_value = model_base.predict(targets.drop(secret_col, axis=1))
            model_base_pred_value = model_base_pred_value[0]
        except Exception as e:
            print(f"A model.predict() error occurred: {e}")
            quit()
        # convert model_base_pred_value to a series
        model_base_pred_value_series = pd.Series(model_base_pred_value, index=targets.index)
        model_base_answer = anonymeter_mods.evaluate_inference_guesses(guesses=model_base_pred_value_series, secrets=targets[secret_col], regression=regression).sum()
        if model_base_answer not in [0,1]:
            print(f"Error: unexpected answer {model_base_answer}")
            sys.exit(1)
        num_model_base_correct += model_base_answer

        # Now run the model attack
        try:
            model_attack_pred_value = model_attack.predict(targets.drop(secret_col, axis=1))
            model_attack_pred_value = model_attack_pred_value[0]
        except Exception as e:
            print(f"A model.predict() error occurred: {e}")
            quit()
        # convert model_attack_pred_value to a series
        model_attack_pred_value_series = pd.Series(model_attack_pred_value, index=targets.index)
        model_attack_answer = anonymeter_mods.evaluate_inference_guesses(guesses=model_attack_pred_value_series, secrets=targets[secret_col], regression=regression).sum()
        if model_attack_answer not in [0,1]:
            print(f"Error: unexpected answer {model_attack_answer}")
            sys.exit(1)
        num_model_attack_correct += model_attack_answer

        # Run the anonymeter-style attack on the synthetic data
        syn_meter_pred_values = []
        syn_meter_pred_value_series = anonymeter_mods.run_anonymeter_attack(
                                        targets=targets,
                                        basis=df_syn[attack_cols],
                                        aux_cols=aux_cols,
                                        secret=secret_col,
                                        regression=regression)
        syn_meter_pred_value = syn_meter_pred_value_series.iloc[0]
        syn_meter_pred_values.append(syn_meter_pred_value)
        syn_meter_answer = anonymeter_mods.evaluate_inference_guesses(guesses=syn_meter_pred_value_series, secrets=targets[secret_col], regression=regression).sum()
        if syn_meter_answer not in [0,1]:
            print(f"Error: unexpected answer {syn_meter_answer}")
            sys.exit(1)
        num_syn_correct += syn_meter_answer

        # Run the anonymeter-style attack on the control data for the baseline
        base_meter_pred_value_series = anonymeter_mods.run_anonymeter_attack(
                                        targets=targets,
                                        basis=df_control[attack_cols],
                                        aux_cols=aux_cols,
                                        secret=secret_col,
                                        regression=regression)
        base_meter_pred_value = base_meter_pred_value_series.iloc[0]
        base_meter_answer = anonymeter_mods.evaluate_inference_guesses(guesses=base_meter_pred_value_series, secrets=targets[secret_col], regression=regression).sum()
        if base_meter_answer not in [0,1]:
            print(f"Error: unexpected answer {base_meter_answer}")
            sys.exit(1)
        num_meter_base_correct += base_meter_answer

        # Now, we want to run the anonymeter-style attack on every valid
        # synthetic dataset. We will use this additional information to decide
        # if the anonymeter-style attack on the full dataset is correct or not.
        num_subset_combs = 0
        num_subset_correct = 0
        col_combs = get_valid_combs(tm, secret_col)
        #print(f"Running with total {max_subsets} of {len(col_combs)} column combinations")
        if len(col_combs) > max_subsets:
            col_combs = random.sample(col_combs, max_subsets)
        pred_values = []
        for col_comb in col_combs:
            df_syn_subset = tm.get_syn_df(col_comb)
            df_syn_subset = convert_datetime_to_timestamp(df_syn_subset)
            df_syn_subset = transform_df_with_update(df_syn_subset, encoders)
            subset_aux_cols = col_comb.copy()
            subset_aux_cols.remove(secret_col)
            subset_meter_pred_value_series = anonymeter_mods.run_anonymeter_attack(
                                            targets=targets[col_comb],
                                            basis=df_syn_subset[col_comb],
                                            aux_cols=subset_aux_cols,
                                            secret=secret_col,
                                            regression=regression)
            subset_meter_pred_value = subset_meter_pred_value_series.iloc[0]
            pred_values.append(subset_meter_pred_value)
            subset_meter_answer = anonymeter_mods.evaluate_inference_guesses(guesses=subset_meter_pred_value_series, secrets=targets[secret_col], regression=regression).sum()
            num_subset_combs += 1
            num_subset_correct += subset_meter_answer

        low_syn_meter_pred_value = find_most_frequent_value(pred_values, 0.5)
        if low_syn_meter_pred_value is not None:
            low_syn_meter_pred_value_series = pd.Series(low_syn_meter_pred_value, index=targets.index)
            low_syn_meter_answer = anonymeter_mods.evaluate_inference_guesses(guesses=low_syn_meter_pred_value_series, secrets=targets[secret_col], regression=regression).sum()
        else:
            low_syn_meter_answer = -1     # no prediction

        high_syn_meter_pred_value = find_most_frequent_value(pred_values, 0.5)
        if high_syn_meter_pred_value is not None:
            high_syn_meter_pred_value_series = pd.Series(high_syn_meter_pred_value, index=targets.index)
            high_syn_meter_answer = anonymeter_mods.evaluate_inference_guesses(guesses=high_syn_meter_pred_value_series, secrets=targets[secret_col], regression=regression).sum()
        else:
            high_syn_meter_answer = -1     # no prediction

        attacks.append({
            'secret_value': str(secret_value),
            'secret_percentage:': secret_percentage,
            'secret_col_type': secret_col_type,
            'model_base_pred_value': str(model_base_pred_value),
            'model_base_answer': str(model_base_answer),
            'model_attack_pred_value': str(model_attack_pred_value),
            'model_attack_answer': str(model_attack_answer),
            'syn_meter_pred_value': str(syn_meter_pred_value),
            'syn_meter_answer': str(syn_meter_answer),
            'high_syn_meter_pred_value': str(high_syn_meter_pred_value),
            'high_syn_meter_answer': str(high_syn_meter_answer),
            'low_syn_meter_pred_value': str(low_syn_meter_pred_value),
            'low_syn_meter_answer': str(low_syn_meter_answer),
            'num_subset_combs': num_subset_combs,
            'num_subset_correct': str(num_subset_correct),
            'base_meter_pred_value': str(base_meter_pred_value),
            'base_meter_answer': str(base_meter_answer),
        })
        #print('---------------------------------------------------')
        #pp.pprint(attacks[-1])
    print(f"num_model_base_correct: {num_model_base_correct}\nnum_syn_correct: {num_syn_correct}\nnum_meter_base_correct: {num_meter_base_correct}\nnum_model_attack_correct: {num_model_attack_correct}")
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
    if tm.orig_meta_data['column_classes'][job['secret']] == 'continuous':
        regression = True
        target_type = 'continuous'
    else:
        regression = False
        target_type = 'categorical'
    attacks = do_inference_attacks(tm, job['secret'], target_type, aux_cols, regression, tm.df_orig, df_control, df_syn, job['num_runs'])
    with open(file_path, 'w') as f:
        json.dump(attacks, f, indent=4)

def gather(instances_path):
    attacks = []
    all_files = list(os.listdir(instances_path))
    # loop through the index and filename of all_files
    for i, filename in enumerate(all_files):
        if not filename.endswith('.json'):
            continue
        with open(os.path.join(instances_path, filename), 'r') as f:
            print(f"Reading {i+1} of {len(all_files)} {filename}")
            res = json.load(f)
            attacks += res
    print(f"Total attacks: {len(attacks)}")

def do_plots():
    pass

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
    if args.command == 'test':
        do_tests()
    elif args.command == 'gather':
        gather(instances_path=os.path.join(attack_path, 'instances'))
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