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
num_tries = 200
num_cols = [20, 40, 80]

def get_precision(noisy_counts, exact_counts):
    print(".", end='', flush=True)
    num_correct = 0
    for i in [0,1]:
        guess = round(sum(noisy_counts[i])/len(noisy_counts[i]))
        if guess == exact_counts[i]:
            num_correct += 1
    return round((num_correct/2)*100)

precision = {}
for c in num_cols:
    ckey = f"{c} cols"
    precision[ckey] = {'1dim': [], '2dim': [], '3dim': []}
    num_correct = [0,0]
    for i in range(num_tries):
        df = pd.DataFrame(np.random.randint(0, 2, size=(200, c)), 
                          columns=[f'col{i}' for i in range(c)])
        cols_without_col1 = [col for col in df.columns if col != 'col1']
        # get the count of each value for col1 in df
        exact_counts = df['col1'].value_counts().tolist()
        noisy_counts = [[],[]]
        df_syn = Synthesizer(df[['col1']]).sample()
        for i in [0,1]:
            noisy_counts[i].append(df_syn[df_syn['col1'] == i].shape[0])
        precision[ckey]['1dim'].append(get_precision(noisy_counts, exact_counts))
        print(f"{i}.1-", end='', flush=True)
        noisy_counts = [[],[]]
        for col in cols_without_col1:
            df_syn = Synthesizer(df[['col1',col]]).sample()
            for i in [0,1]:
                noisy_counts[i].append(df_syn[df_syn['col1'] == i].shape[0])
        precision[ckey]['2dim'].append(get_precision(noisy_counts, exact_counts))
        print(f"{i}.2-", end='', flush=True)
        noisy_counts = [[],[]]
        for comb in itertools.combinations(cols_without_col1, 2):
            cols = ['col1'] + list(comb)
            df_syn = Synthesizer(df[cols]).sample()
            for i in [0,1]:
                noisy_counts[i].append(df_syn[df_syn['col1'] == i].shape[0])
        precision[ckey]['3dim'].append(get_precision(noisy_counts, exact_counts))
        print(f"{i}.3-", end='', flush=True)
    precision[ckey]['1dim'] = statistics.mean(precision[ckey]['1dim'])
    precision[ckey]['2dim'] = statistics.mean(precision[ckey]['2dim'])
    precision[ckey]['3dim'] = statistics.mean(precision[ckey]['3dim'])
    print(precision)
    # dump precision as a json file
    with open('exact_count_precision.json', 'w') as f:
        json.dump(precision, f, indent=4)