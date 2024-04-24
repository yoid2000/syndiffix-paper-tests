import sys
import pandas as pd
import numpy as np
import statistics
import itertools
import json
from syndiffix import Synthesizer
from syndiffix.common import AnonymizationParams
import os

'''
This program tests the ability for an attacker to determine the exact count of
a given column value.

The idea is that, by taking many samples of the column value count over many
different synthetic datasets, the attacker can average out the noise. 

In this test, we test averaging multiple 2dim and 3dim synthetic datasets.

This test assumes the best case for the attacker, which is that no combinations
of the target column value with other column values are suppressed. To do this, we
have only two values per column, and 200 rows. This ensures that, for 3dim synthetic
datasets, we have 200/8=25 expected rows per combination, which is well above the
suppression limit
'''

TEST = False
if TEST is False:
    from syndiffix_tools.tree_walker import *
num_other_rows = 50
num_cols = [20, 40, 80, 160]
num_vals = [2, 4, 8]
dims = [1,2,3]
num_rows = [10, 20, 40]
# We need to allow the base_rows_per_val to randomly vary in order to avoid
# bias due to rounding effects when adjusting counts
base_rows_per_val_min = 17
base_rows_per_val_max = 23
# We want more runs for lower dimension data because each run has fewer samples
# We need more runs to get significant results (3-column measures take a long time
# to run)
max_samples = 20000
min_samples = 40
TWO_COLS = True
THREE_COLS = True
attack_keys = ['1dim', '2dim', '3dim']

# read command line arguments
do_attack_num = None
make_slurm = False
if len(sys.argv) > 1:
    if sys.argv[1] == 'slurm':
        make_slurm = True
    else:
        do_attack_num = int(sys.argv[1])

def print_progress_wheel(wheel):
    print(next(wheel) + '\b', end='', flush=True)

def progress_wheel():
    wheel = itertools.cycle(['-', '/', '|', '\\'])
    while True:
        yield next(wheel)

def get_precision(noisy_counts, exact_count, true_row_count):
    num_correct = 0
    # true_row_count assumes that we can determine the exact number of
    # rows by taking average from all tables
    guess = round(sum(noisy_counts)/len(noisy_counts))
    if guess == exact_count:
        correct = 1
    else:
        correct = 0
    error = abs(guess - exact_count)
    return {'correct': correct, 'guessed': guess, 'exact': exact_count, 'error': error}

def make_df(num_val, num_col, num_row, this_try, seed):
    np.random.seed(seed)
    data = {}
    base_rows_per_val = np.random.randint(base_rows_per_val_min, base_rows_per_val_max)
    num_rows_base = base_rows_per_val * num_val
    num_rows_total = num_rows_base + num_row

    # Create a custom probability distribution
    prob_dist = np.linspace(1, 1.3, num_val)
    prob_dist /= prob_dist.sum()

    for i in range(num_col):
        if i == 0:
            # The 0th column has the value 1
            data[f'col{this_try}_{i}'] = [1] * num_rows_base
            # Now add the target rows, with value 0
            for j in range(num_row):
                data[f'col{this_try}_{i}'].append(0)
        else:
            # The remaining columns each have num_val distinct values, assigned according to the custom probability distribution
            values = np.random.choice(range(num_val), num_rows_total, p=prob_dist)
            data[f'col{this_try}_{i}'] = values
    df = pd.DataFrame(data)
    df = df.sample(frac=1).reset_index(drop=True)
    return df

def get_col_combs(df, col0, dim):
    cols_without_col0 = [col for col in df.columns if col != col0]
    col_combs = []
    if dim == 1:
        col_combs.append([col0])
    if dim == 2:
        for col in cols_without_col0:
            col_combs.append([col0,col])
    elif dim == 3:
        for comb in itertools.combinations(cols_without_col0, 2):
            col_combs.append([col0] + list(comb))
    return col_combs

def do_attack(num_val, num_col, dim, num_row):
    '''
    num_val: number of distinct values in each other column
    num_col: number of columns
    dim: number of dimensions of the attack
    num_row: number of rows of the value being predicted
    '''
    file_name = f'v{num_val}.c{num_col}.d{dim}.r{num_row}.json'
    if TEST is True: print(file_name)
    file_path = os.path.join('exact_count_results', file_name)
    if TEST is False and os.path.exists(file_path):
        return
    prec = {
        'num_val': num_val,
        'num_col': num_col,
        'dim': dim,
        'num_row': num_row,
        'correct_averages': 0,
        'error_averages': 0,
        'error_std_devs': 0,
        'samples': 0,
        'total_table_rows': 0,
        'errors': [],
        'scores': [],
        'results': [],
        'tree_walks': {}
    }
    if dim == 1:
        num_tries = max_samples
    elif dim == 2:
        num_tries = int(max(min_samples, max_samples / (num_col - 1)))
    elif dim == 3:
        num_tries = int(max(min_samples, max_samples / (((num_col-1) * (num_col-2)) / 2)))
    if TEST is True:
        num_tries = 2
    for this_try in range(num_tries):
        # set the seed for np.random
        seed = this_try + (num_col * 100) + (num_val * 1000) + (num_row * 10000)
        if TEST is True: 
            print(seed)
        df = make_df(num_val, num_col, num_row, this_try, seed)
        if TEST is True:
            for tcol in df.columns:
                print('---', df[tcol].value_counts())
            continue
        prec['total_table_rows'] = df.shape[0]
        col0 = f'col{this_try}_0'
        exact_count = df[df[col0] == 0].shape[0]
        col_combs = get_col_combs(df, col0, dim)
        # get the count of the target value 0 for col0 in df
        if TEST: print(col_combs)
        noisy_counts = []
        for col_comb in col_combs:
            # Set the SynDiffix salt to avoid the same noise across different experimantal settings
            sdx_seed = str(seed).encode()
            syn = Synthesizer(df[col_comb],
                    anonymization_params=AnonymizationParams(salt=sdx_seed))
            df_syn = syn.sample()
            ncount = df_syn[df_syn[col0] == 0].shape[0]
            noisy_counts.append(ncount)
            error = abs(ncount - exact_count)
            if TEST is False and error not in prec['tree_walks']:
                tw = TreeWalker(syn)
                prec['tree_walks'][error] = tw.get_forest_nodes()
        result = get_precision(noisy_counts, exact_count, df.shape[0])
        prec['results'].append(result)
        prec['scores'].append(result['correct'])
        prec['errors'].append(result['error'])
    if TEST: return
    prec['correct_averages'] = statistics.mean(prec['scores'])
    prec['error_averages'] = statistics.mean(prec['errors'])
    if len(prec['errors']) > 1:
        prec['error_std_devs'] = statistics.stdev(prec['errors'])
    else:
        prec['error_std_devs'] = 0
    prec['samples'] = len(prec['scores'])
    # dump precision as a json file
    # make directory 'exact_count_results' if it does not exist
    if not os.path.exists('exact_count_results'):
        os.makedirs('exact_count_results')
    with open(file_path, 'w') as f:
        json.dump(prec, f, indent=4)

wheel = progress_wheel()
attack_num = 0
for num_val in num_vals:
    for num_col in num_cols:
        for dim in dims:
            for num_row in num_rows:
                if make_slurm is False and (do_attack_num is None or attack_num == do_attack_num):
                    do_attack(num_val, num_col, dim, num_row)
                attack_num += 1
if make_slurm:
    with open('exact_count_slurm.sh', 'w') as f:
        f.write('#!/bin/bash\n')
        f.write('#SBATCH --time=0-24:00\n')
        f.write('#SBATCH --mem=10G\n')
        f.write('#SBATCH --output /dev/null\n')
        f.write('#SBATCH --array=0-' + str(attack_num - 1) + '\n')
        f.write('source ../sdx_tests/sdx_venv/bin/activate' + '\n')
        f.write('python exact_count.py $SLURM_ARRAY_TASK_ID\n')