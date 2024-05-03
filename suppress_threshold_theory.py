import argparse
import os
import pandas as pd
import numpy as np
import json
from syndiffix import Synthesizer
from syndiffix.common import AnonymizationParams, SuppressionParams
import pprint
import sys

if 'SDX_TEST_DIR' in os.environ:
    base_path = os.getenv('SDX_TEST_DIR')
else:
    base_path = os.getcwd()
if 'SDX_TEST_CODE' in os.environ:
    code_path = os.getenv('SDX_TEST_CODE')
else:
    code_path = None
os.makedirs(base_path, exist_ok=True)
runs_path = os.path.join(base_path, 'suppress_theory')
os.makedirs(runs_path, exist_ok=True)
tests_path = os.path.join(runs_path, 'tests')
os.makedirs(tests_path, exist_ok=True)
results_path = os.path.join(tests_path, 'results')
os.makedirs(results_path, exist_ok=True)
pp = pprint.PrettyPrinter(indent=4)

def summarize_stats(stats):
    summary = {
        'neg_pos_1_neg_1': 0,
        'neg_pos_1_neg_0': 0,
        'neg_pos_0_neg_1': 0,
        'neg_pos_0_neg_0': 0,
        'pos_pos_1_neg_1': 0,
        'pos_pos_1_neg_0': 0,
        'pos_pos_0_neg_1': 0,
        'pos_pos_0_neg_0': 0,
    }
    for thing in stats:
        if thing['case'] == 'positive':
            if thing['pos_signal'] > 0:
                if thing['neg_signal'] > 0:
                    summary['pos_pos_1_neg_1'] += 1
                else:
                    summary['pos_pos_1_neg_0'] += 1
            else:
                if thing['neg_signal'] > 0:
                    summary['pos_pos_0_neg_1'] += 1
                else:
                    summary['pos_pos_0_neg_0'] += 1
        else:
            if thing['pos_signal'] > 0:
                if thing['neg_signal'] > 0:
                    summary['neg_pos_1_neg_1'] += 1
                else:
                    summary['neg_pos_1_neg_0'] += 1
            else:
                if thing['neg_signal'] > 0:
                    summary['neg_pos_0_neg_1'] += 1
                else:
                    summary['neg_pos_0_neg_0'] += 1
    return summary

def gather_results():
    json_files = [pos_json for pos_json in os.listdir(tests_path) if pos_json.endswith('.json')]

    data_list = []

    for file in json_files:
        file_path = os.path.join(tests_path, file)
        with open(file_path, "r") as json_file:
            print(f"Reading {file_path}")
            data = json.load(json_file)
            data_dict = {key: data[key] for key in ("tp", "fp", "tn", "fn", "rows_mult", "num_target_val", "low_mean_gap", "samples", "dim")}
            data_dict['summary'] = summarize_stats(data['stats'])
            data_list.append(data_dict)
    # make a path to suppress_threshold_results.json in directory results
    json_path = os.path.join(results_path, 'suppress_threshold_results.json')
    # Dump results as a json file
    print(f"Writing results to {json_path}")
    with open(json_path, 'w') as f:
        json.dump(data_list, f, indent=4)

def make_plot():
    import matplotlib.pyplot as plt
    import matplotlib.lines as mlines
    import seaborn as sns
    # read file suppress_threshold_results.json
    supp_res_path = os.path.join(results_path, 'suppress_threshold_results.json')
    with open(supp_res_path, 'r') as f:
        results = json.load(f)

    mults = {'5 rows_multiplier': 5,
            '10 rows_multiplier': 10,
            '100 rows_multiplier': 100}

    tars = {'2 target_vals': 2,
            '5 target_vals': 5,
            '10 target_vals': 10}

    gaps = {'2.0 low_mean_gap': 2.0,
            '3.0 low_mean_gap': 3.0,
            '4.0 low_mean_gap': 4.0}

    data = []
    for multk, multv in results.items():
        mult = mults[multk]
        for tark, targ in multv.items():
            tar = tars[tark]
            for gapk, gapv in targ.items():
                gap = gaps[gapk]
                # This is the ratio of the number of rows with a given c1
                # to the number of rows with the victim's c1 value.
                # 3 is the number of other c1 vals
                target_size = (gapv['num_rows'] / tar) / 3
                # This would be the TP rate if we made a statistical guess
                mean = 3 + gap
                stat_guess = 1 / tar
                tp_rate = max(1/30000, gapv['tp'] / gapv['samples'])
                fp_rate = max(1/30000, gapv['fp'] / gapv['samples'])
                tn_rate = max(1/30000, gapv['tn'] / gapv['samples'])
                fn_rate = max(1/30000, gapv['fn'] / gapv['samples'])
                all_pos = gapv['tp'] + gapv['fp']
                if all_pos == 0.0:
                    precision = 0
                    precision_improvement = 0.0
                else:
                    precision = gapv['tp'] / all_pos
                    precision_improvement = (precision - stat_guess) / (1.0 - stat_guess) 
                coverage = all_pos / gapv['samples']
                # A value of 0 would screw up the log scale
                coverage = max(1/30000, coverage)
                data.append({
                    'sd_gap': int(gap),
                    'mult': int(mult),
                    'num_targets': int(tar),
                    'mean': mean,
                    'precision': precision,
                    'precision_improvement': precision_improvement,
                    'coverage': coverage,
                    'target_size': int(target_size),
                    'stat_guess': stat_guess,
                    'tp_rate': tp_rate,
                    'fp_rate': fp_rate,
                    'tn_rate': tn_rate,
                    'fn_rate': fn_rate,
                    'samples': gapv['samples'],
                    'num_rows': gapv['num_rows'],
                    'tp': gapv['tp'],
                    'fp': gapv['fp'],
                    'tn': gapv['tn'],
                    'fn': gapv['fn'],
                })

    df = pd.DataFrame(data)
    print(df[['sd_gap','sd_gap','num_targets','target_size']].to_string())

    # Define the marker styles for sd_gap
    marker_map = {2: 'o', 3: '^', 4: 'v'}

    # Define the color map for target_size
    colors = sns.color_palette()[:3]
    color_map = {5: colors[0], 10: colors[1], 100: colors[2]}

    # Define the size map for num_targets
    size_map = {2: 30, 5: 100, 10: 180}
    legend_size_map = {2: 60, 5: 90, 10: 120}

    # Create the scatter plot
    plt.figure(figsize=(7, 3.5))

    for (sd_gap, marker) in marker_map.items():
        for (target_size, color) in color_map.items():
            for (num_targets, size) in size_map.items():
                df_filtered = df[(df['sd_gap'] == sd_gap) & (df['target_size'] == target_size) & (df['num_targets'] == num_targets)]
                print(sd_gap, target_size, num_targets)
                print(df_filtered.to_string())
                plt.scatter(df_filtered['coverage'], df_filtered['precision_improvement'], color=color, marker=marker, s=size, alpha=0.6)

    # Add horizontal lines
    plt.axhline(0, color='black', linestyle='--')
    plt.axhline(0.5, color='black', linestyle='--')

    # Set axis labels
    plt.xscale('log')
    plt.xlabel('Coverage (log scale)', fontsize=13, labelpad=10)
    plt.ylabel('Precision Improvement', fontsize=13, labelpad=10)

    # Create legends
    legend1 = plt.legend([mlines.Line2D([0], [0], color='black', marker=marker, linestyle='None') for sd_gap, marker in marker_map.items()], ['sd_gap: {}'.format(sd_gap) for sd_gap in marker_map.keys()], title='', loc='lower left', bbox_to_anchor=(0.23, 0), fontsize='small')
    legend2 = plt.legend([mlines.Line2D([0], [0], color=color, marker='o', linestyle='None') for target_size, color in color_map.items()], ['target_size: {}'.format(target_size) for target_size in color_map.keys()], title='', loc='lower left', bbox_to_anchor=(0.43, 0), fontsize='small')
    plt.legend([mlines.Line2D([0], [0], color='black', marker='o', markersize=size/10, linestyle='None') for num_targets, size in legend_size_map.items()], ['num_targets: {}'.format(num_targets) for num_targets in legend_size_map.keys()], title='', loc='lower left', bbox_to_anchor=(0.70, 0), fontsize='small')

    plt.gca().add_artist(legend1)
    plt.gca().add_artist(legend2)

    # Modify x-axis ticks and labels
    ticks = list(plt.xticks()[0]) + [1/30000]
    labels = [t if t != 1/30000 else '<1/30k' for t in ticks]
    plt.xticks(ticks, labels)

    # Set x-axis range to min and max 'coverage' values
    plt.xlim(1/(30000 + 5000), 0.02)

    # Create the path to suppress.png
    path_to_suppress_png = os.path.join(results_path, 'suppress.png')

    # Save the plot as a PNG file
    plt.savefig(path_to_suppress_png, dpi=300, bbox_inches='tight')

    plt.close()

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
    data[f'col{x}_1'] = np.random.randint(0, num_target_val, num_rows)

    # Create columns col_2 through col_C-1 with two distinct integer values, each chosen randomly from integers between 0 and 1000
    k, l = np.random.randint(0, 1001, 2)
    for i in range(2, num_cols):
        data[f'col{x}_{i}'] = np.random.choice([k,l], num_rows)

    df = pd.DataFrame(data)

    # Create the attack conditions
    df.loc[0, f'col{x}_0'] = 'z'
    df.loc[0, f'col{x}_1'] = 0
    df.loc[1, f'col{x}_0'] = 'z'
    df.loc[1, f'col{x}_1'] = 0
    df.loc[2, f'col{x}_0'] = 'z'
    target_val = df.loc[2, f'col{x}_1']
    # Need to shuffle the dataframes otherwise we'll get the same
    # noise due to the same indices assigned by syndiffix
    df = df.sample(frac=1).reset_index(drop=True)
    return df, target_val

def _run_attack(x, file_name):
    file_path = os.path.join(tests_path, file_name)
    # use this to record results of positive and negative cases
    x['stats'] = []
    # Compute num_rows such that there are not many suppressed combinations
    for this_try in range(x['samples']):
        if this_try % 1000 == 999:
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
        for comb in combs:
            syn = Synthesizer(df,
                anonymization_params=AnonymizationParams(low_count_params=SuppressionParams(low_mean_gap=x['low_mean_gap'])))
            df_syn = syn.sample()
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
    num_tries_by_lmg = [30000, 30000, 30000]
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
                results['samples'] = num_tries_by_lmg[i]
                for dim in dims:
                    if dim == 2 and low_mean_gap != 2.0:
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
    else:
        try:
            job_num = int(args.command)
            run_attack(job_num=job_num)
        except ValueError:
            print(f"Unknown command: {args.command}")

if __name__ == "__main__":
    main()