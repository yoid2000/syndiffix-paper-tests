from anonymeter.evaluators import InferenceEvaluator
import argparse
import os
import pandas as pd
import json
import sys
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
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
num_attacks = 20000

def make_config():
    ''' I want to generate num_attacks attacks. Each attack will be on a given secret
    column in a given table. I will run multiple of these attacks per secret/table if
    necessary.
    '''
    # Initialize attack_jobs
    attack_jobs = []

    # Loop over each directory name in syn_path
    while len(attack_jobs) < num_attacks:
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
                })
    # randomize the order in which the attack_jobs are run
    random.shuffle(attack_jobs)
    for index, job in enumerate(attack_jobs):
        job['index'] = index
    # remove any extra attack_jobs
    attack_jobs = attack_jobs[:num_attacks]
    for index, job in enumerate(attack_jobs):
        job['index'] = index
        print(index, job)

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
    with open(os.path.join(attack_path, 'attack.slurm'), 'w', encoding='utf-8') as f:
        f.write(slurm_template)

def do_inference_attack(secret, aux_cols, regression, df_original, df_control, df_syn):
    ''' df_original and df_control have all columns.
        df_syn has only the columns in aux_cols and secret.
    '''
    attack_cols = aux_cols + [secret]
    # Call the evaluator with only the attack_cols, because I'm not sure if it will
    # work if different dataframes have different columns
    evaluator = InferenceEvaluator(ori=df_original[attack_cols],
                                    syn=df_syn[attack_cols],
                                    control=df_control[attack_cols],
                                    aux_cols=aux_cols,
                                    secret=secret,
                                    regression=regression,
                                    n_attacks=1)
    evaluator.evaluate(n_jobs=-2)
    return evaluator

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
    # set aux_cols to all columns except the secret column
    aux_cols = [col for col in df_syn.columns if col not in job['secret']]
    if tm.orig_meta_data['column_classes'][job['secret']] == 'continuous':
        regression = True
    else:
        regression = False
    evaluator = do_inference_attack(job['secret'], aux_cols, regression, tm.df_orig, df_control, df_syn)
    evalRes = evaluator.results()
    print("Successs rate of main attack:", evalRes.attack_rate)
    print("Successs rate of baseline attack:", evalRes.baseline_rate)
    print("Successs rate of control attack:", evalRes.control_rate)
    privRisk = evalRes.risk()
    print(privRisk)
    print("Queries:", evaluator.queries())
    pass

def gather(instances_path):
    pass

def do_plots():
    pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", help="'config' to run make_config(), or an integer to run run_attacks()")
    args = parser.parse_args()

    if args.command == 'config':
        make_config()
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