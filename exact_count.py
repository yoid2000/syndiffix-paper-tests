import pandas as pd
import numpy as np
import statistics
import itertools
import json
from syndiffix import Synthesizer

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

num_tries = 1000
num_rows = 300
num_cols = [20, 40, 80]
# We want more runs for lower dimension data because each run has fewer samples
# We need more runs to get significant results (3-column measures take a long time
# to run)
runs_per_num_col = [10000, 1000, 100]
TWO_COLS = True
THREE_COLS = True

def print_progress_wheel(wheel):
    print(next(wheel) + '\b', end='', flush=True)

def progress_wheel():
    wheel = itertools.cycle(['-', '/', '|', '\\'])
    while True:
        yield next(wheel)

def get_precision(noisy_counts, exact_counts):
    num_correct = 0
    for i in [0,1]:
        guess = round(sum(noisy_counts[i])/len(noisy_counts[i]))
        if guess == exact_counts[i]:
            num_correct += 1
    return round((num_correct/2)*100)

wheel = progress_wheel()
precision = {}
for c in num_cols:
    ckey = f"{c} cols"
    precision[ckey] = {'1dim': [], '2dim': [], '3dim': []}
    num_correct = [0,0]
    for this_try in range(max(runs_per_num_col)):
        df = pd.DataFrame(np.random.randint(0, 2, size=(num_rows, c)), 
                          columns=[f'col{this_try}_{i}' for i in range(c)])
        col0 = f'col{this_try}_0'
        cols_without_col0 = [col for col in df.columns if col != col0]
        # get the count of each value for col0 in df
        exact_counts = df[col0].value_counts().tolist()
        noisy_counts = [[],[]]
        df_syn = Synthesizer(df[[col0]]).sample()
        for i in [0,1]:
            noisy_counts[i].append(df_syn[df_syn[col0] == i].shape[0])
        precision[ckey]['1dim'].append(get_precision(noisy_counts, exact_counts))
        print(f"{c}-{this_try}.1", flush=True)
        noisy_counts = [[],[]]
        if TWO_COLS and this_try <= runs_per_num_col[1]:
            for col in cols_without_col0:
                df_syn = Synthesizer(df[[col0,col]]).sample()
                print_progress_wheel(wheel)
                for i in [0,1]:
                    noisy_counts[i].append(df_syn[df_syn[col0] == i].shape[0])
            precision[ckey]['2dim'].append(get_precision(noisy_counts, exact_counts))
            print(f"{c}-{this_try}.2", flush=True)
        if THREE_COLS and this_try <= runs_per_num_col[2]:
            noisy_counts = [[],[]]
            for comb in itertools.combinations(cols_without_col0, 2):
                cols = [col0] + list(comb)
                df_syn = Synthesizer(df[cols]).sample()
                print_progress_wheel(wheel)
                for i in [0,1]:
                    noisy_counts[i].append(df_syn[df_syn[col0] == i].shape[0])
            precision[ckey]['3dim'].append(get_precision(noisy_counts, exact_counts))
            print(f"{c}-{this_try}.3", flush=True)
    precision[ckey]['1dim'] = statistics.mean(precision[ckey]['1dim'])
    if TWO_COLS:
        precision[ckey]['2dim'] = statistics.mean(precision[ckey]['2dim'])
    if THREE_COLS:
        precision[ckey]['3dim'] = statistics.mean(precision[ckey]['3dim'])
    print(precision)
    # dump precision as a json file
    with open('exact_count_precision.json', 'w') as f:
        json.dump(precision, f, indent=4)