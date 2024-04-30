import argparse
import os
import pandas as pd
import json
import glob
from pathlib import Path


def make_config():
    # Read the environment variable SDX_TEST_DIR and assign it to base_path
    base_path = os.getenv('SDX_TEST_DIR')
    code_path = os.getenv('SDX_TEST_CODE')

    # Create pq_path and attack_path
    pq_path = os.path.join(base_path, 'original_data_parquet')
    attack_path = os.path.join(base_path, 'suppress_attacks')

    # Create directory at attack_path if it doesn't exist
    os.makedirs(attack_path, exist_ok=True)

    # Initialize attack_jobs
    attack_jobs = []

    # Loop over each .parquet file in pq_path
    for file_path in glob.glob(os.path.join(pq_path, '*.parquet')):
        # Read the file into a DataFrame
        df = pd.read_parquet(file_path)

        # Get the file base name without the .parquet suffix
        file_base = os.path.basename(file_path)[:-len('.parquet')]

        # Loop over each column in df
        for column in df.columns:
            # Create a dict and append it to attack_jobs
            attack_jobs.append({
                'index': len(attack_jobs),
                'file_base': file_base,
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

def run_attack(integer):
    # Your code here
    pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", help="'config' to run make_config(), or an integer to run run_attack()")
    args = parser.parse_args()

    if args.command == 'config':
        make_config()
    else:
        try:
            integer = int(args.command)
            run_attack(integer)
        except ValueError:
            print(f"Unknown command: {args.command}")

if __name__ == "__main__":
    main()