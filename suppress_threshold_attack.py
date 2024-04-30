import argparse
import os
import pandas as pd
import json
import glob
from syndiffix_tools.tables_manager import TablesManager


def make_config():
    # Read the environment variable SDX_TEST_DIR and assign it to base_path
    base_path = os.getenv('SDX_TEST_DIR')
    code_path = os.getenv('SDX_TEST_CODE')

    # Create pq_path and attack_path
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
    # write the slurm template to a file suppress_threshold_attack.slurm
    with open(os.path.join(attack_path, 'suppress_threshold_attack.slurm'), 'w') as f:
        f.write(slurm_template)

def run_attack(job_num):
    # Your code here
    base_path = os.getenv('SDX_TEST_DIR')
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