import pandas as pd
import numpy as np
import statistics
import itertools
import json
from syndiffix import Synthesizer
from syndiffix.common import AnonymizationParams, SuppressionParams

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
'''
num_tries = 200
'''
class SuppressionParams:
    low_threshold: int = 3
    layer_sd: float = 1.0
    low_mean_gap: float = 2.0

NOISELESS_SUPPRESSION = SuppressionParams(layer_sd=0.0)

NOISELESS_PARAMS = AnonymizationParams(
    low_count_params=NOISELESS_SUPPRESSION,
    layer_noise_sd=0.0,
    outlier_count=FlatteningInterval(upper=FlatteningInterval().lower),
    top_count=FlatteningInterval(upper=FlatteningInterval().lower),
)
syn_data = Synthesizer(raw_data, anonymization_params=NOISELESS_PARAMS).sample()
'''

Synthesizer(df_original,
            anonymization_params=AnonymizationParams(low_count_params=SuppressionParams(low_threshold=3, layer_sd=1.0, low_mean_gap=2.0)))


def get_precision(noisy_counts, exact_counts):
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
    for this_try in range(num_tries):
        df = pd.DataFrame(np.random.randint(0, 2, size=(200, c)), 
                          columns=[f'col{this_try}_{i}' for i in range(c)])
        col1 = f'col{this_try}_1'
        cols_without_col1 = [col for col in df.columns if col != col1]
        # get the count of each value for col1 in df
        exact_counts = df[col1].value_counts().tolist()
        noisy_counts = [[],[]]
        df_syn = Synthesizer(df[[col1]]).sample()
        for i in [0,1]:
            noisy_counts[i].append(df_syn[df_syn[col1] == i].shape[0])
        precision[ckey]['1dim'].append(get_precision(noisy_counts, exact_counts))
        print(f"{this_try}.1-", end='', flush=True)
        noisy_counts = [[],[]]
        for col in cols_without_col1:
            df_syn = Synthesizer(df[[col1,col]]).sample()
            print(".", end='', flush=True)
            for i in [0,1]:
                noisy_counts[i].append(df_syn[df_syn[col1] == i].shape[0])
        precision[ckey]['2dim'].append(get_precision(noisy_counts, exact_counts))
        print(f"{this_try}.2-", end='', flush=True)
        noisy_counts = [[],[]]
        for comb in itertools.combinations(cols_without_col1, 2):
            cols = [col1] + list(comb)
            df_syn = Synthesizer(df[cols]).sample()
            print(".", end='', flush=True)
            for i in [0,1]:
                noisy_counts[i].append(df_syn[df_syn[col1] == i].shape[0])
        precision[ckey]['3dim'].append(get_precision(noisy_counts, exact_counts))
        print(f"{this_try}.3-", end='', flush=True)
    precision[ckey]['1dim'] = statistics.mean(precision[ckey]['1dim'])
    precision[ckey]['2dim'] = statistics.mean(precision[ckey]['2dim'])
    precision[ckey]['3dim'] = statistics.mean(precision[ckey]['3dim'])
    print(precision)
    # dump precision as a json file
    with open('exact_count_precision.json', 'w') as f:
        json.dump(precision, f, indent=4)