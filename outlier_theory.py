import argparse
import os
import pandas as pd
import numpy as np
import json
from syndiffix import Synthesizer
from syndiffix.common import AnonymizationParams, SuppressionParams
# from syndiffix_tools.tree_walker import *
import pprint
import sys

'''
All outlier tests are based on detecting whether there are:
        0 or 1 outlier
        1 or 2 outliers
        2 or 3 outliers
Value outlier tests:
    Detect presence:
        We know the presence of the outlier will raise the average value
        of the column with the outlier value.
    Infer value of another column:
'''


save_results = False
if 'SDX_TEST_DIR' in os.environ:
    base_path = os.getenv('SDX_TEST_DIR')
else:
    base_path = os.getcwd()
if 'SDX_TEST_CODE' in os.environ:
    code_path = os.getenv('SDX_TEST_CODE')
else:
    code_path = None
os.makedirs(base_path, exist_ok=True)
runs_path = os.path.join(base_path, 'outlier_theory')
os.makedirs(runs_path, exist_ok=True)
tests_path = os.path.join(runs_path, 'tests')
os.makedirs(tests_path, exist_ok=True)
results_path = os.path.join(runs_path, 'results')
os.makedirs(results_path, exist_ok=True)
pp = pprint.PrettyPrinter(indent=4)

def make_slurm():
    exe_path = os.path.join(code_path, 'suppress_threshold_theory.py')
    venv_path = os.path.join(base_path, 'sdx_venv', 'bin', 'activate')
    slurm_dir = os.path.join(runs_path, 'slurm_out')
    os.makedirs(slurm_dir, exist_ok=True)
    slurm_out = os.path.join(slurm_dir, 'out.%a.out')
    num_jobs = run_attack(count_jobs=True)
    # Define the slurm template
    slurm_template = f'''#!/bin/bash
#SBATCH --job-name=suppress_theory
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
    with open(os.path.join(runs_path, 'theory.slurm'), 'w') as f:
        f.write(slurm_template)
import pandas as pd
import numpy as np

def make_df(col1_vals, num_rows, num_target_val, dim):
    num_rows += 3   # for the attack conditions
    num_cols = dim + 2     # the additional attack columns plus the attack conditions
    # Initialize the data dictionary
    x = np.random.randint(0, 1000000)
    data = {}

    # Create col_0 with 'a', 'b', 'c' distributed uniformly randomly
    data[f'col{x}_0'] = np.random.choice(col1_vals, num_rows)

    # Create col_1 with num_target_val distinct integer values from 0 to num_target_val-1, distributed uniformly randomly
    data[f'col{x}_1'] = np.random.randint(int(known_target_val), num_target_val+int(known_target_val), num_rows)

    # Create columns col_2 through col_C-1 with two distinct integer values, each chosen randomly from integers between 200 and 1200
    k, l = np.random.randint(200, 1201, 2)
    for i in range(2, num_cols):
        data[f'col{x}_{i}'] = np.random.choice([k,l], num_rows)

    df = pd.DataFrame(data)

    # Create the attack conditions
    df.loc[0, f'col{x}_0'] = 'z'
    df.loc[0, f'col{x}_1'] = int(known_target_val)
    df.loc[1, f'col{x}_0'] = 'z'
    df.loc[1, f'col{x}_1'] = int(known_target_val)
    df.loc[2, f'col{x}_0'] = 'z'
    target_val = df.loc[2, f'col{x}_1']
    # Need to shuffle the dataframes otherwise we'll get the same
    # noise due to the same indices assigned by syndiffix
    df = df.sample(frac=1).reset_index(drop=True)
    return df, target_val

def dump_and_exit(df, df_syn, forest):
    print("Original")
    print(df.to_string())
    print("Synthetic")
    print(df_syn.to_string())
    print("Forest")
    pp.pprint(forest)
    sys.exit(1)

def check_for_target_nodes_consistency(df, df_syn, forest, c0, c1, c0_supp, c0_c1_supp_target, c0_c1_supp_victim):
    '''
    We're interested in two nodes where 'z' might show up. One is in the 1dim tree for column c0, and the other is in the 2dim tree for columns c0/c1. We want to make sure that the nodes are consistently suppressed or not suppressed.

    c0 and c1 are the column names
    v0 is the target value ('z')
    c0_supp and c0_c1_supp are the suppression status of prior walks (or None)
    '''
    for node in forest.values():
        if len(node['columns']) == 1 and node['columns'][0] == c0 and node['actual_intervals'][0] == [3.0, 3.0]:
            # This is my 1dim node of interest
            if node['true_count'] != 3:
                print(f"Error: 1dim node has count {node['true_count']}")
                dump_and_exit(df, df_syn, forest)
            if c0_supp is None:
                c0_supp = node['over_threshold']
            elif c0_supp != node['over_threshold']:
                print(f"Error: 1dim node has inconsistent suppression")
                dump_and_exit(df, df_syn, forest)
        if len(node['columns']) == 2 and node['columns'] == [c0, c1] and node['actual_intervals'][0] == [3.0, 3.0] and node['singularity'] is True:
            # This is one of my 2dim nodes of interest
            if node['true_count'] >= 2:
                if node['actual_intervals'][1] != [known_target_val, known_target_val]:
                    # This must be the known persons node, and so the target
                    # value must be known_target_val
                    print(f"Error: 2dim target node should have value 0")
                    dump_and_exit(df, df_syn, forest)
                if node['true_count'] == 2:
                    if c0_c1_supp_target is None:
                        c0_c1_supp_target = node['over_threshold']
                    elif c0_c1_supp_target != node['over_threshold']:
                        print(f"Error: 2dim target node has inconsistent suppression")
                        dump_and_exit(df, df_syn, forest)
            if node['true_count'] == 1:
                if node['actual_intervals'][1] == [known_target_val, known_target_val]:
                    # This must be the victim's node, and so the 
                    # target value must not be known_target_val
                    print(f"Error: 2dim victim node should not have value 0")
                    dump_and_exit(df, df_syn, forest)
                if c0_c1_supp_victim is None:
                    c0_c1_supp_victim = node['over_threshold']
                elif c0_c1_supp_victim != node['over_threshold']:
                    print(f"Error: 2dim victim node has inconsistent suppression")
                    dump_and_exit(df, df_syn, forest)
    return c0_supp, c0_c1_supp_target, c0_c1_supp_victim

def _run_attack(x, file_name):
    file_path = os.path.join(tests_path, file_name)
    # use this to record results of positive and negative cases
    x['stats'] = []
    # Compute num_rows such that there are not many suppressed combinations
    for this_try in range(x['samples']):
        if this_try % 1000 == 999 and save_results:
            # dump the results as json to file_path
            with open(file_path, 'w') as f:
                json.dump(x, f, indent=4)
        # Use different column names with each run so as to get different noise
        col1_vals = ['a', 'b', 'c']
        x['num_rows'] = len(col1_vals) * x['num_target_val'] * x['rows_mult']
        df, target_val = make_df(col1_vals, x['num_rows'], x['num_target_val'], x['dim'])
        c0 = df.columns[0]
        c1 = df.columns[1]

        # We loop through the basic (2dim) attack and all of the
        # 3dim attacks
        combs = [[c0, c1]]
        for i in range(2, len(df.columns)):
            combs.append([c0, c1, df.columns[i]])
        num_combs_with_z_and_0 = 0
        num_combs_with_z_and_not_0 = 0
        c0_supp = None
        c0_c1_supp_target = None
        c0_c1_supp_victim = None
        for comb in combs:
            syn = Synthesizer(df[comb],
                anonymization_params=AnonymizationParams(low_count_params=SuppressionParams(low_mean_gap=x['low_mean_gap'])))
            df_syn = syn.sample()
            if len(combs) > 1:
                tw = TreeWalker(syn)
                forest = tw.get_forest_nodes()
                c0_supp, c0_c1_supp_target, c0_c1_supp_victim = check_for_target_nodes_consistency(df, df_syn, forest, c0, c1, c0_supp, c0_c1_supp_target, c0_c1_supp_victim)
            num_rows_with_z_and_not_0 = len(df_syn[(df_syn[c0] == 'z') & (df_syn[c1] != 0)])
            if num_rows_with_z_and_not_0 > 0:
                num_combs_with_z_and_not_0 += 1
            num_rows_with_z_and_0 = len(df_syn[(df_syn[c0] == 'z') & (df_syn[c1] == 0)])
            if num_rows_with_z_and_0 > 0:
                num_combs_with_z_and_0 += 1
        if target_val == 0:
            x['stats'].append({'case': 'positive',
                         'pos_signal': num_combs_with_z_and_0, 
                         'neg_signal': num_combs_with_z_and_not_0})
        else:
            x['stats'].append({'case': 'negative',
                         'pos_signal': num_combs_with_z_and_0, 
                         'neg_signal': num_combs_with_z_and_not_0})
        if num_combs_with_z_and_not_0 == 0 and num_combs_with_z_and_0 > 0:
            # positive guess
            if target_val == 0:
                # correct
                x['tp'] += 1
            else:
                # wrong
                x['fp'] += 1
        else:
            # negative guess
            if target_val == 0:
                # wrong
                x['fn'] += 1
            else:
                # correct
                x['tn'] += 1
    if save_results:
        with open(file_path, 'w') as f:
            json.dump(x, f, indent=4)

def run_attack(job_num=None, count_jobs=False):
    '''
    This program tests the ability of an attacker to determine if a victim has
    a given attribute by observing whether suppression has taken place.

    The conditions for the attack are as follows:
    * The attacker knows that there are exactly three persons with a given attribute
    A (i.e. a column and associated value).
    * The attacker knows that two of them definately also have attribute B.

    The goal is to determine whether or not the third person (the victim)
    also has attribute B.  Because SynDiffix suppresses aggregates with two users,
    but may not suppress an aggregate with three users, if a row with attributes A
    and B are in the synthetic dataset (i.e. not suppressed), then the victim must
    have attribute B.

    We test several different values of the low_mean_gap.

    We test two different table parameters. One parameter is num_target_vals. This
    is the number of distinct target attribute (i.e. B) values that exist. This
    parameter effects the baseline statistical prediction that an attacker can make.
    For instance, if there are 5 target values, then a prediction that the victim
    has the target value has a precision of 20%.

    The second table parameter is rows_multiplier. 
    '''
    '''
    class SuppressionParams:
        low_threshold: int = 3
        layer_sd: float = 1.0
        low_mean_gap: float = 2.0

    The 1dim node with count 3 containing the victim can certainly be non-suppressed.
    '''
    low_mean_gaps = [2.0, 3.0, 4.0]
    num_target_vals = [2, 5, 10]
    rows_multiplier = [5, 10, 100]
    dims = [20, 0]
    num_jobs = 0

    results = {}
    results = {'tp':0 , 'fp':0, 'tn':0, 'fn':0}
    for rows_mult in rows_multiplier:
        results['rows_mult'] = rows_mult
        for num_target_val in num_target_vals:
            results['num_target_val'] = num_target_val
            for i in range(len(low_mean_gaps)):
                low_mean_gap = low_mean_gaps[i]
                results['low_mean_gap'] = low_mean_gap
                results['samples'] = num_tries
                for dim in dims:
                    if dim == 0 and low_mean_gap != 2.0:
                        continue
                    if save_results is False and dim != 20:
                        continue
                    results['dim'] = dim
                    if count_jobs is False and (job_num is None or num_jobs == job_num):
                        results['job_num'] = num_jobs
                        file_name = f"res.rm{rows_mult}.tv{num_target_val}.lmg{low_mean_gap}.dim{dim}.json"
                        print(f"Running attack with rows_mult={rows_mult}, num_target_val={num_target_val}, low_mean_gap={low_mean_gap}, dim={dim}")
                        print(f"Results will be saved to {file_name}")
                        _run_attack(results, file_name)
                    if count_jobs is True:
                        print(f"Job {num_jobs} will run attack with rows_mult={rows_mult}, num_target_val={num_target_val}, low_mean_gap={low_mean_gap}, dim={dim}")
                    num_jobs += 1
    if count_jobs is True:
        return num_jobs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", help="'slurm' to make slurmscript, 'plot' to plot the results, 'attacks' to run all attacks, or an integer to run a specific attack")
    args = parser.parse_args()

    if args.command == 'slurm':
        make_slurm()
    elif args.command == 'plot':
        make_plot()
    elif args.command == 'attacks':
        run_attack()
    elif args.command == 'gather':
        gather_results()
    elif args.command == 'membership':
        membership_attack()
    else:
        try:
            job_num = int(args.command)
            run_attack(job_num=job_num)
        except ValueError:
            print(f"Unknown command: {args.command}")

if __name__ == "__main__":
    main()