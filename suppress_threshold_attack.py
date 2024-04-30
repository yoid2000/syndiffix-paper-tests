import argparse
import os
import pandas as pd
import json
from syndiffix_tools.tables_manager import TablesManager


def make_config():
    # Read the environment variable SDX_TEST_DIR and assign it to base_path
    base_path = os.getenv('SDX_TEST_DIR')
    code_path = os.getenv('SDX_TEST_CODE')

    # Create syn_path and attack_path
    syn_path = os.path.join(base_path, 'synDatasets')
    attack_path = os.path.join(base_path, 'suppress_attacks')

    # Create directory at attack_path if it doesn't exist
    os.makedirs(attack_path, exist_ok=True)

    # Initialize attack_jobs
    attack_jobs = []

    # Loop over each directory name in syn_path
    for dir_name in os.listdir(syn_path):
        # Create a TablesManager object with the directory path
        tm = TablesManager(dir_path=os.path.join(syn_path, dir_name))

        # Get the original DataFrame columns
        columns = list(tm.df_orig.columns)

        # Get the protected ID columns
        pid_cols = tm.get_pid_cols()
        if len(pid_cols) > 0:
            # We can't really run the attack on time-series data
            continue

        # Remove the protected ID columns from columns
        columns = [col for col in columns if col not in pid_cols]

        # Loop over each column in columns
        for column in columns:
            # Create a dict and append it to attack_jobs
            attack_jobs.append({
                'index': len(attack_jobs),
                'dir_name': dir_name,
                'column': column
            })

    # Write attack_jobs into a JSON file
    with open(os.path.join(attack_path, 'attack_jobs.json'), 'w') as f:
        json.dump(attack_jobs, f, indent=4)

    exe_path = os.path.join(code_path, 'suppress_threshold_attack.py')
    venv_path = os.path.join(base_path, 'sdx_venv', 'bin', 'activate')
    slurm_out = os.path.join(attack_path, 'slurm_out')
    os.makedirs(slurm_out, exist_ok=True)
    num_jobs = len(attack_jobs) - 1
    # Define the slurm template
    slurm_template = f'''#!/bin/bash
#SBATCH --job-name=suppress_attack
#SBATCH --output={slurm_out}
#SBATCH --error={slurm_out}
#SBATCH --time=7-0
#SBATCH --mem=16G
#SBATCH --cpus-per-task=1
#SBATCH --array=0-{num_jobs}
source {venv_path}
python {exe_path} $array
'''
    # write the slurm template to a file attack.slurm
    with open(os.path.join(attack_path, 'attack.slurm'), 'w') as f:
        f.write(slurm_template)

def make_attack_setup(tm, file_path, job):
    attack_setup = {'setup': {}, 'attack_instances': []}

    # Find all values that appear exactly 3 times in tm.df_orig
    known_column = job['column']
    value_counts = tm.df_orig[known_column].value_counts()
    known_vals = value_counts[value_counts == 3].index.tolist()

    # Set setup values
    attack_setup['setup']['num_rows'] = len(tm.df_orig)
    attack_setup['setup']['job'] = job

    for known_val in known_vals:
        # Find all columns where at least two of the 3 rows have the same value
        known_rows = tm.df_orig[tm.df_orig[known_column] == known_val]
        for col in known_rows.columns:
            if col == known_column:
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
                attack_instance = {
                    'target_col': target_col,
                    'target_val': target_val,
                    'victim_val': victim_val,
                    'known_col': known_column,
                    'kwown_val': known_val,
                    'correct_pred': correct_pred,
                    'file_path': file_path,
                    'known_rows': known_rows.to_dict(orient='records'),
                    'num_target_vals': tm.df_orig[target_col].nunique()
                }
                attack_setup['attack_instances'].append(attack_instance)
    attack_setup['setup']['num_instances'] = len(attack_setup['attack_instances'])

    # Write attack_setup to file_path
    with open(file_path, 'w') as f:
        json.dump(attack_setup, f, indent=4)

    return attack_setup

def run_attack(job_num):
    # Your code here
    base_path = os.getenv('SDX_TEST_DIR')

    # Create syn_path and attack_path
    syn_path = os.path.join(base_path, 'synDatasets')
    attack_path = os.path.join(base_path, 'suppress_attacks')

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
    file_name = f"{job['column']}_{job['dir_name']}.json"
    file_path = os.path.join(instances_path, file_name)

    # Make a TablesManager object
    tm = TablesManager(dir_path=os.path.join(syn_path, job['dir_name']))

    # If file_path exists, read it into attack_setup. Otherwise, call make_attack_setup()
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            attack_setup = json.load(f)
    else:
        attack_setup = make_attack_setup(tm, file_path, job)

    pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", help="'config' to run make_config(), or an integer to run run_attack()")
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