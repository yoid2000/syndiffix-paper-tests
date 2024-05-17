import argparse
import os
import pandas as pd
import json
import sys
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, auc
import matplotlib.pyplot as plt
import itertools
import pprint

pp = pprint.PrettyPrinter(indent=4)

remove_bad_files = False
#sample_for_model = 200000
sample_for_model = None
roll_window = 5000
do_comb_3_and_4 = False

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
attack_path = os.path.join(base_path, 'suppress_attacks')
os.makedirs(attack_path, exist_ok=True)
max_attacks = 200000

def do_model():
    # Read in the parquet file
    model_stats = {}
    res_path = os.path.join(attack_path, 'results.parquet')
    df = pd.read_parquet(res_path)

    if sample_for_model is not None:
        df = df.sample(n=sample_for_model, random_state=42)

    # Convert 'c' column to binary
    df['c'] = df['c'].map({'positive': 1, 'negative': 0})

    # Separate features and target
    X = df.drop(columns=['c'])
    y = df['c']

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

    # Retain a copy of X_test which includes all columns
    X_test_all = X_test.copy()

    unneeded_columns = ['cap', 'capt', 'tp']
    # Standardize the features
    scaler = StandardScaler()
    # Scale the data
    columns = X_train.drop(columns=unneeded_columns).columns
    X_train_scaled = scaler.fit_transform(X_train.drop(columns=unneeded_columns))
    X_train = pd.DataFrame(X_train_scaled, columns=columns)

    columns = X_test.drop(columns=unneeded_columns).columns
    X_test_scaled = scaler.transform(X_test.drop(columns=unneeded_columns))
    X_test = pd.DataFrame(X_test_scaled, columns=columns)
    print(f"X_train type is {type(X_train)}, y_train type is {type(y_train)}")

    # Train the model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Get the feature importance
    importance = model.coef_[0]
    # Map feature numbers to names
    feature_names = X_train.columns
    feature_importance = pd.DataFrame({'Feature': feature_names, 'Importance': importance})

    # Sort by absolute value of importance
    feature_importance['abs_importance'] = feature_importance['Importance'].abs()
    feature_importance = feature_importance.sort_values(by='abs_importance', ascending=False)
    print(feature_importance[['Feature', 'Importance']])

    # save feature_importance as a dictionary
    model_stats['feature_importance'] = feature_importance.set_index('Feature')['Importance'].to_dict()

    # Get the probability of positive class
    y_score = model.predict_proba(X_test)[:,1]

    # Add y_score into the retained copy as an additional column
    X_test_all['prob_tp'] = y_score

    # Save X_test_all, y_test, and y_score to parquet files
    X_test_all.to_parquet(os.path.join(attack_path, 'X_test.parquet'))
    pd.DataFrame(y_score, columns=['prob_tp']).to_parquet(os.path.join(attack_path, 'y_score.parquet'))
    pd.DataFrame(y_test).to_parquet(os.path.join(attack_path, 'y_test.parquet'))

    # write model_stats to json file
    with open(os.path.join(attack_path, 'model_stats.json'), 'w') as f:
        json.dump(model_stats, f, indent=4)

def do_plots():
    # Read in the parquet files
    X_test_all = pd.read_parquet(os.path.join(attack_path, 'X_test.parquet'))
    y_test = pd.read_parquet(os.path.join(attack_path, 'y_test.parquet')).squeeze()
    y_score = pd.read_parquet(os.path.join(attack_path, 'y_score.parquet')).squeeze()

    X_test_all['pi'] = (X_test_all['prob_tp'] - X_test_all['frac_tar']) / (1.00001 - X_test_all['frac_tar'])

    pi_floor = 0
    X_test_all['pi_fl'] = X_test_all['pi'].clip(lower=pi_floor)

    # print distributions
    print("Distribution of capt:")
    print(X_test_all['capt'].describe())
    print("Distribution of cap:")
    print(X_test_all['cap'].describe())
    print("Distribution of pi_fl:")
    print(X_test_all['pi_fl'].describe())
    print("Distribution of bs:")
    print(X_test_all['bs'].describe())
    avg_capt = X_test_all['capt'].mean()
    print(f"Average capt: {avg_capt}")
    avg_cap = X_test_all['cap'].mean()
    print(f"Average cap: {avg_cap}")

    print("X_test:")
    print(X_test_all.head())
    print(f"Total rows: {X_test_all.shape[0]}")

    # Generate a dataframe that bins pi_fl with 20 bins of equal width
    df_bin = pd.cut(X_test_all['pi_fl'], bins=20)
    df_bin['pi_fl_mid'] = df_bin.apply(lambda x: x.mid)
    df_bin['count'] = df_bin.groupby(bins)['pi_fl'].transform('count')
    df_bin['frac_perfect'] = df_bin['count'] / df_bin['count'].sum()
    print(df_bin.head())
    quit()

    if False:
        # Compute precision-recall curve
        #precision, recall, _ = precision_recall_curve(y_test, y_score)
        precision, recall, _ = precision_recall_curve(y_test, X_test_all['prob_tp'])
        # Plot precision-recall curve
        plt.figure()
        plt.plot(recall, precision, color='darkorange', lw=2, label='PR curve')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall curve')
        plt.legend(loc="lower right")
        plot_path = os.path.join(attack_path, 'pr_curve.png')
        plt.savefig(plot_path)

    # Sort the DataFrame by 'pi_fl' in descending order and reset the index
    X_test_all_sorted = X_test_all.sort_values(by='pi_fl', ascending=False).reset_index(drop=True)

    X_test_all_sorted['prob_perfect'] = (X_test_all_sorted.index + 1) / len(X_test_all_sorted)
    X_test_all_sorted['prob_combs_targets'] = X_test_all_sorted['prob_perfect'] * avg_capt
    X_test_all_sorted['prob_combs'] = X_test_all_sorted['prob_perfect'] * avg_cap
    df_plot = X_test_all_sorted.reset_index(drop=True)
    # Reverse the DataFrame
    df_plot_roll = df_plot.rolling(window=roll_window).mean()

    # Plot 'probability' vs 'pi_fl'
    plt.figure(figsize=(6, 3))
    plt.plot(df_plot['prob_perfect'], df_plot['pi_fl'], label='Attack conditions exist')
    plt.plot(df_plot['prob_combs'], df_plot['pi_fl'], label='Attacker knowledge, any target ok')
    plt.plot(df_plot['prob_combs_targets'], df_plot['pi_fl'], label='Attacker knowledge, only specific target')
    plt.xscale('log')
    plt.hlines(0.5, 0.001, 1, colors='black', linestyles='--', linewidth=0.5)
    plt.vlines(0.001, 0.5, 1.0, colors='black', linestyles='--', linewidth=0.5)
    plt.xlabel('Coverage (log)')
    plt.ylabel(f'Precision Improvement\n(floored at {pi_floor})')
    plt.legend(loc="lower left", prop={'size': 8})
    plt.tight_layout()
    plt.savefig(os.path.join(attack_path, 'pi_cov.png'))
    plt.close()

    # Plot 'probability' vs 'pi_fl' with rolling average
    plt.figure(figsize=(6, 3))
    plt.plot(df_plot_roll['prob_perfect'], df_plot_roll['pi_fl'], label='Attack conditions exist')
    plt.plot(df_plot_roll['prob_combs'], df_plot_roll['pi_fl'], label='Attacker knowledge, any target ok')
    plt.plot(df_plot_roll['prob_combs_targets'], df_plot_roll['pi_fl'], label='Attacker knowledge, only specific target')
    plt.xscale('log')
    plt.hlines(0.5, 0.001, 1, colors='black', linestyles='--', linewidth=0.5)
    plt.vlines(0.001, 0.5, 1.0, colors='black', linestyles='--', linewidth=0.5)
    plt.xlabel(f'Coverage (log), rolling average (window={roll_window})')
    plt.ylabel(f'Precision Improvement\n(floored at {pi_floor})')
    plt.legend(loc="lower left", prop={'size': 8})
    plt.tight_layout()
    plt.savefig(os.path.join(attack_path, 'pi_cov_roll.png'))
    plt.close()

def gather(instances_path):
    all_entries = []
    
    num_fail = 0
    # Step 1: Read in all of the json files in the directory at instances_path
    all_files = list(os.listdir(instances_path))
    # loop through the index and filename of all_files
    for i, filename in enumerate(all_files):
        if filename.endswith('.json'):
            with open(os.path.join(instances_path, filename), 'r') as f:
                print(f"Reading {i+1} of {len(all_files)} {filename}")
                file_length = os.path.getsize(f.name)
                if file_length < 10:
                    # Can happen if the file is still under construction
                    print(f"---- File {filename} is too short, skipping")
                    continue
                try:
                    res = json.load(f)
                    capt = res['summary']['coverage_all_combs_targets']
                    cap = res['summary']['coverage_all_combs']
                    num_rows = res['summary']['num_rows']
                    for entry in res['attack_results']:
                        entry['capt'] = capt
                        entry['cap'] = cap
                        entry['frac_tar'] = entry['nrtv'] / num_rows
                        all_entries.append(entry)
                except json.JSONDecodeError:
                    num_fail += 1
                    if remove_bad_files:
                        print(f"---- Error reading {filename}, deleting")
                        os.remove(os.path.join(instances_path, filename))
                    else:
                        print(f"---- Error reading {filename}, skipping")
                    continue
                
    # Step 4: Make a dataframe df where each key in each entry of all_entries is a column
    df = pd.DataFrame(all_entries)
    print(df.head())
    
    # Step 5: Store df as a parquet file called results.parquet
    file_path = os.path.join(attack_path, 'results.parquet')
    df.to_parquet(file_path)
    print(f"{num_fail} files were corrupted and deleted")

def make_config():
    # Initialize attack_jobs
    attack_jobs = []

    # Loop over each directory name in syn_path
    for dir_name in os.listdir(syn_path):
        # Create a TablesManager object with the directory path
        tm = TablesManager(dir_path=os.path.join(syn_path, dir_name))
        columns = list(tm.df_orig.columns)
        # Get the protected ID columns
        pid_cols = tm.get_pid_cols()
        if len(pid_cols) > 0:
            # We can't really run the attack on time-series data
            continue
        # Remove the protected ID columns from columns
        columns = [col for col in columns if col not in pid_cols]
        for col1, col2 in itertools.combinations(columns, 2):
            attack_jobs.append({
                'index': len(attack_jobs),
                'dir_name': dir_name,
                'columns': [col1, col2],
            })
    # randomize the order in which the attack_jobs are run
    random.shuffle(attack_jobs)

    # Write attack_jobs into a JSON file
    with open(os.path.join(attack_path, 'attack_jobs.json'), 'w') as f:
        json.dump(attack_jobs, f, indent=4)

    exe_path = os.path.join(code_path, 'suppress_threshold_attack.py')
    venv_path = os.path.join(base_path, 'sdx_venv', 'bin', 'activate')
    slurm_dir = os.path.join(attack_path, 'slurm_out')
    os.makedirs(slurm_dir, exist_ok=True)
    slurm_out = os.path.join(slurm_dir, 'out.%a.out')
    num_jobs = len(attack_jobs) - 1
    # Define the slurm template
    slurm_template = f'''#!/bin/bash
#SBATCH --job-name=suppress_attack
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
    with open(os.path.join(attack_path, 'attack.slurm'), 'w') as f:
        f.write(slurm_template)

def to_list(value):
    if isinstance(value, str) or not hasattr(value, '__iter__'):
        return [value]
    else:
        return list(value)

def get_attack_info(df_syn, comb, known_val_comb, target_col, target_val):
    ''' Find the synthetic data that correpsonds to the columns in comb+target_col.
        This might be the full synthetic table (for instance, if comb has 3 columns).
        Prior to this, we have already determined that there are either 2 or 3 rows
        that contain the known_val_comb values in the columns in comb. We have also
        determined that 2 of those rows have value target_val in target_col. We want
        to see if the corresponding rows appear in the synthetic data.
    '''
    if df_syn is None:
        print(f"Could not find a synthetic table for columns {comb}")
        sys.exit(1)
    if len(list(df_syn.columns)) == len(comb+[target_col]):
        best_syn = True
    else:
        best_syn = False
    mask = (df_syn[comb] == known_val_comb).all(axis=1)
    subset = df_syn[mask]
    mask_with_target = subset[target_col] == target_val
    num_with_target = int(mask_with_target.sum())
    mask_without_target = subset[target_col] != target_val
    num_without_target = int(mask_without_target.sum())
    return num_with_target, num_without_target, best_syn

def run_attacks(tm, file_path, job):
    max_attack_instances = 10
    attack_summary = {'summary': {'num_samples':[0,0,0,0,0],
                              'num_possible_combs_targets': 0,
                              'num_possible_combs': 0,
                              'num_rows': tm.df_orig.shape[0],
                              'num_cols': tm.df_orig.shape[1],
                              'num_attacks': 0,
                              'tp': 0,
                              'fp': 0,
                              'tn': 0,
                              'fn': 0,
                              'finished': True,
                              'job':job},
                       'sample_instances': [[],[],[],[],[]],
                       'attack_results': []}

    columns = list(tm.df_orig.columns)
    if do_comb_3_and_4:
        combinations = list(itertools.combinations(columns, 3)) + list(itertools.combinations(columns, 4))
        combinations = [comb for comb in combinations if all(col in comb for col in job['columns'])]
        combs = [[job['columns'][0]], [job['columns'][1]], job['columns']] + combinations
    else:
        combs = [[job['columns'][0]], [job['columns'][1]], job['columns']]

    # For each comb, find all values that appear exactly 3 times in tm.df_orig
    sum_base_probs = 0
    do_summarize = False
    for comb in combs:
        if len(attack_summary['attack_results']) >= max_attacks:
            attack_summary['summary']['finished'] = False
            break
        if do_summarize:
            summarize_and_write(attack_summary, file_path, sum_base_probs)
            do_summarize = False
        if len(comb) == len(columns):
            # Can't have a target unknown column in this case
            continue
        comb = list(comb)
        num_known_columns = len(comb)
        # Group the DataFrame by the columns in comb and count the number of rows for each group
        grouped = tm.df_orig.groupby(comb).size()
        attack_summary['summary']['num_possible_combs'] += grouped.shape[0]
        attack_summary['summary']['num_possible_combs_targets'] += grouped.shape[0] * (tm.df_orig.shape[1] - num_known_columns)

        # Filter the groups to only include those with exactly 3 rows
        known_val_combs = grouped[grouped == 3].index.tolist()
        for col in list(tm.df_orig.columns):
            # We loop through the columns first so that we only need to pull in the
            # relevant df_syn once per col
            if col in comb:
                continue
            df_syn = None
            for known_val_comb in known_val_combs:
                # Find all columns where at least two of the 3 rows have the same value
                known_val_comb = to_list(known_val_comb)
                mask = (tm.df_orig[comb] == known_val_comb).all(axis=1)
                known_rows = tm.df_orig[mask]
                target_col = col
                target_val = None
                if known_rows[col].nunique() == 1:
                    # set target_val to the mode of known_rows[col]
                    target_val = known_rows[col].mode()[0]
                    victim_val = target_val
                    correct_pred = 'positive'
                elif known_rows[col].nunique() == 2:
                    # set target_val to the mode value of known_rows[col]
                    target_val = known_rows[col].mode()[0]
                    # set victim_val to the other value
                    victim_val = known_rows[col][known_rows[col] != target_val].values[0]
                    correct_pred = 'negative'
                if target_val is None:
                    continue
                num_rows_with_target_val = len(tm.df_orig[tm.df_orig[target_col] == target_val])
                num_distinct_values = len(tm.df_orig[target_col].unique())
                attack_summary['summary']['num_samples'][num_known_columns] += 1
                if len(attack_summary['sample_instances'][num_known_columns]) < max_attack_instances:
                    # The stringify's are needed to avoid json serialization issues
                    attack_instance = {
                        'target_col': target_col,
                        'num_rows_with_target_val': num_rows_with_target_val,
                        'num_distinct_target_col_vals': num_distinct_values,
                        'target_val': str(target_val),
                        'victim_val': str(victim_val),
                        'known_cols': comb,
                        'known_vals': known_val_comb,
                        'correct_pred': correct_pred,
                        'file_path': file_path,
                        #'known_rows': str(known_rows.to_dict(orient='records')),
                    }
                    attack_summary['sample_instances'][num_known_columns].append(attack_instance)
                if df_syn is None:
                    df_syn = tm.get_best_syn_df(columns=comb+[target_col])
                num_with_target, num_without_target, best_syn = get_attack_info(df_syn, comb, known_val_comb, target_col, target_val)
                got_tp = False
                # probability that simply guessing positive would be correct
                sum_base_probs += num_rows_with_target_val / tm.df_orig.shape[0]
                if correct_pred == 'positive' and num_with_target > 0 and num_without_target == 0:
                    attack_summary['summary']['tp'] += 1
                    got_tp = True
                elif correct_pred == 'negative' and num_with_target > 0 and num_without_target == 0:
                    attack_summary['summary']['fp'] += 1
                elif correct_pred == 'positive' and (num_with_target == 0 or num_without_target > 0):
                    attack_summary['summary']['fn'] += 1
                elif correct_pred == 'negative' and (num_with_target == 0 or num_without_target > 0):
                    attack_summary['summary']['tn'] += 1
                attack_result = {
                    # number rows with target value
                    'nrtv': num_rows_with_target_val,
                    # number of distinct target values
                    'ndtv': num_distinct_values,
                    # What a correct prediction would be
                    'c': correct_pred,
                    # number of synthetic rows with knwon values and target value
                    'nkwt': num_with_target,
                    # number of synthetic rows with known values and not target value
                    'nkwot': num_without_target,
                    # whether the synthetic table is the best one
                    'bs': best_syn,
                    # the number of known columns
                    'nkc': num_known_columns,
                    # whether simple critieria yielded true positive
                    'tp': got_tp,
                }
                attack_summary['attack_results'].append(attack_result)
                if len(attack_summary['attack_results']) % 10000 == 0:
                    do_summarize = True
            df_syn = None
    summarize_and_write(attack_summary, file_path, sum_base_probs)

def summarize_and_write(attack_summary, file_path, sum_base_probs):
    tp = attack_summary['summary']['tp']
    fp = attack_summary['summary']['fp']
    tn = attack_summary['summary']['tn']
    fn = attack_summary['summary']['fn']
    num_attacks = tp + fp + tn + fn
    avg_base = sum_base_probs / num_attacks if num_attacks > 0 else 0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    pi = (prec - avg_base) / (1 - avg_base) if (1 - avg_base) != 0 else 0
    if (tp + fp + tn + fn) != len(attack_summary['attack_results']):
        print(f"Error: {tp} + {fp} + {tn} + {fn} != {len(attack_summary['attack_results'])}")
        sys.exit(1)
    attack_summary['summary']['num_attacks'] = num_attacks
    attack_summary['summary']['precision'] = prec
    attack_summary['summary']['precision_improvement'] = pi
    attack_summary['summary']['coverage_known_naive'] = (tp + fp) / num_attacks if num_attacks != 0 else 0
    attack_summary['summary']['coverage_all_combs'] = (tp + fp) / attack_summary['summary']['num_possible_combs'] if attack_summary['summary']['num_possible_combs'] != 0 else 0
    attack_summary['summary']['coverage_all_combs_targets'] = (tp + fp) / attack_summary['summary']['num_possible_combs_targets'] if attack_summary['summary']['num_possible_combs_targets'] != 0 else 0

    # Write attack_summary to file_path
    with open(file_path, 'w') as f:
        json.dump(attack_summary, f, indent=4)

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
    column_str = '_'.join(job['columns'])
    file_name = f"{job['dir_name']}.{column_str}.json"
    file_path = os.path.join(instances_path, file_name)

    if os.path.exists(file_path):
        print(f"File already exists: {file_path}")
        return

    # Make a TablesManager object
    tm = TablesManager(dir_path=os.path.join(syn_path, job['dir_name']))
    run_attacks(tm, file_path, job)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", help="'config' to run make_config(), or an integer to run run_attacks()")
    args = parser.parse_args()

    if args.command == 'config':
        make_config()
    elif args.command == 'gather':
        gather(instances_path=os.path.join(attack_path, 'instances'))
    elif args.command == 'model':
        do_model()
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