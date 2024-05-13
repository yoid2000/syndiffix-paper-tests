import argparse
import os
import pandas as pd
import json
import sys
from syndiffix_tools.tables_manager import TablesManager
import itertools

base_path = os.getenv('SDX_TEST_DIR')
code_path = os.getenv('SDX_TEST_CODE')
syn_path = os.path.join(base_path, 'synDatasets')
attack_path = os.path.join(base_path, 'suppress_attacks')
os.makedirs(attack_path, exist_ok=True)

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

def get_attack_info(tm, comb, known_val_comb, target_col, target_val):
    ''' Find the synthetic data that correpsonds to the columns in comb+target_col.
        This might be the full synthetic table (for instance, if comb has 3 columns).
        Prior to this, we have already determined that there are either 2 or 3 rows
        that contain the known_val_comb values in the columns in comb. We have also
        determined that 2 of those rows have value target_val in target_col. We want
        to see if the corresponding rows appear in the synthetic data.
    '''
    df_syn = tm.get_best_syn_df(columns=comb+[target_col])
    if df_syn is None:
        print(f"Could not find a synthetic table for columns {comb}")
        sys.exit(1)
    if len(list(df_syn.columns)) == len(comb):
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
    max_attack_instances = 100
    attack_summary = {'summary': {'num_samples':[0,0,0,0,0],
                              'num_possible_known_value_combs': 0,
                              'num_rows': tm.df_orig.shape[0],
                              'num_attacks': 0,
                              'job':job},
                       'sample_instances': [[],[],[],[],[]],
                       'attack_results': []}

    columns = list(tm.df_orig.columns)
    combinations = list(itertools.combinations(columns, 3)) + list(itertools.combinations(columns, 4))
    combinations = [comb for comb in combinations if all(col in comb for col in job['columns'])]
    combs = [[job['columns'][0]], [job['columns'][1]], job['columns']] + combinations

    # For each comb, find all values that appear exactly 3 times in tm.df_orig
    for comb in combs:
        if len(comb) == len(columns):
            # Can't have a target unknown column in this case
            continue
        comb = list(comb)
        num_known_columns = len(comb)
        # Group the DataFrame by the columns in comb and count the number of rows for each group
        grouped = tm.df_orig.groupby(comb).size()
        # This is the total number of 
        attack_summary['summary']['num_possible_known_value_combs'] += grouped.shape[0]

        # Filter the groups to only include those with exactly 3 rows
        known_val_combs = grouped[grouped == 3].index.tolist()
        got_instance_sample = False
        for known_val_comb in known_val_combs:
            # Find all columns where at least two of the 3 rows have the same value
            known_val_comb = to_list(known_val_comb)
            mask = (tm.df_orig[comb] == known_val_comb).all(axis=1)
            known_rows = tm.df_orig[mask]
            for col in known_rows.columns:
                if col in comb:
                    continue
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
                if target_val is not None:
                    num_rows_with_target_val = len(tm.df_orig[tm.df_orig[target_col] == target_val])
                    num_distinct_values = len(tm.df_orig[target_col].unique())
                    attack_summary['summary']['num_samples'][num_known_columns] += 1
                    if len(attack_summary['sample_instances'][num_known_columns]) < max_attack_instances and not got_instance_sample:
                        got_instance_sample = True
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
                            'known_rows': known_rows.to_dict(orient='records'),
                        }
                        attack_summary['sample_instances'][num_known_columns].append(attack_instance)
                    num_with_target, num_without_target, best_syn = get_attack_info(tm, comb, known_val_comb, target_col, target_val)
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
                    }
                    attack_summary['attack_results'].append(attack_result)
    attack_summary['summary']['num_attacks'] = len(attack_summary['attack_results'])

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

    # Make a TablesManager object
    tm = TablesManager(dir_path=os.path.join(syn_path, job['dir_name']))
    run_attacks(tm, file_path, job)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", help="'config' to run make_config(), or an integer to run run_attacks()")
    args = parser.parse_args()

    if args.command == 'config':
        make_config()
    else:
        try:
            job_num = int(args.command)
            run_attack(job_num)
        except ValueError:
            print(f"Unknown command: {args.command}")

if __name__ == "__main__":
    main()