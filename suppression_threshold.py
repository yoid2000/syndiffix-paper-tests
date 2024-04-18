import pandas as pd
import numpy as np
import json
import pprint
from syndiffix import Synthesizer
from syndiffix.common import AnonymizationParams, SuppressionParams

'''
This program tests the ability of an attacker to determine if a victim is
in a given table or not.

The conditions for the attack are as follows:
* The attacker knows that there are exactly either two or three persons with a given attribute

The goal is to determine whether or not the third person (the victim)
is in the table.  Because SynDiffix suppresses aggregates with two users,
but may not suppress an aggregate with three users, if any rows have the
given attribute, then the victim must be in the table.
'''

pp = pprint.PrettyPrinter(indent=4)
low_mean_gaps = [2.0, 3.0, 4.0]
num_tries_by_lmg = [4000, 20000, 40000]

results = {}
for i in range(len(low_mean_gaps)):
    low_mean_gap = low_mean_gaps[i]
    lmg_key = f'{low_mean_gap} low_mean_gap'
    num_tries = num_tries_by_lmg[i]
    col1_vals = ['a', 'b', 'c']
    # Compute num_rows such that there are not many suppressed combinations
    num_rows = 75
    results[lmg_key] = {'tp':0, 'fp':0, 'tn':0, 'fn':0, 'samples': num_tries, 'num_rows': num_rows}
    for this_try in range(num_tries):
        # Use different column names with each run so as to get different noise
        c1 = f"c1_{this_try}"
        df = pd.DataFrame({c1: np.random.choice(col1_vals, size=num_rows),})
        # Add two rows of the attack configuration
        df = pd.concat([df, pd.DataFrame({c1: ['z']})], ignore_index=True)
        df = pd.concat([df, pd.DataFrame({c1: ['z']})], ignore_index=True)
        
        # add the third row with 50% probability
        if np.random.randint(0, 2) == 1:
            df = pd.concat([df, pd.DataFrame({c1: ['z']})], ignore_index=True)
            status = 'positive'
        else:
            status = 'negative'

        # Need to shuffle the dataframes otherwise we'll get the same
        # noise due to the same indices assigned by syndiffix
        df = df.sample(frac=1).reset_index(drop=True)

        syn = Synthesizer(df,
            anonymization_params=AnonymizationParams(low_count_params=SuppressionParams(low_mean_gap=low_mean_gap)))
        df_syn = syn.sample()
        num_rows_with_z = len(df_syn[(df_syn[c1] == 'z')])
        if num_rows_with_z > 0:
            # positive guess
            if status == 'positive':
                # correct
                results[lmg_key]['tp'] += 1
            else:
                # wrong
                results[lmg_key]['fp'] += 1
        else:
            # negative guess
            if status == 'positive':
                # wrong
                results[lmg_key]['fn'] += 1
            else:
                # correct
                results[lmg_key]['tn'] += 1
    coverage = (results[lmg_key]['tp'] + results[lmg_key]['fp']) / results[lmg_key]['samples']
    results[lmg_key]['coverage'] = coverage

    print(results)
# Dump results as a json file
with open('suppress_threshold_results_no_infer.json', 'w') as f:
    json.dump(results, f, indent=4)