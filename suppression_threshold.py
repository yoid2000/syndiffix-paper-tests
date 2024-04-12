import pandas as pd
import numpy as np
import json
import pprint
from syndiffix import Synthesizer
from syndiffix.common import AnonymizationParams, SuppressionParams
from syndiffix_tools.tree_walker import *

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
pp = pprint.PrettyPrinter(indent=4)
low_mean_gaps = [2.0, 3.0, 4.0]
num_target_vals = [2, 5, 10]
rows_multiplier = [5, 10, 100]
num_tries_by_lmg = [1000, 5000, 10000]

results = {}
for rows_mult in rows_multiplier:
    rm_key = f'{rows_mult} rows_multiplier'
    results[rm_key] = {}
    for num_target_val in num_target_vals:
        ntv_key = f'{num_target_val} target_vals'
        results[rm_key][ntv_key] = {}
        for i in range(len(low_mean_gaps)):
            low_mean_gap = low_mean_gaps[i]
            lmg_key = f'{low_mean_gap} low_mean_gap'
            num_tries = num_tries_by_lmg[i]
            col1_vals = ['a', 'b', 'c']
            # Compute num_rows such that there are not many suppressed combinations
            num_rows = len(col1_vals) * num_target_val * rows_mult
            results[rm_key][ntv_key][lmg_key] = {'tp':0, 'fp':0, 'tn':0, 'fn':0, 'samples': num_tries, 'num_rows': num_rows}
            for this_try in range(num_tries):
                # Use different column names with each run so as to get different noise
                c1 = f"c1_{this_try}"
                c2 = f"c2_{this_try}"
                df = pd.DataFrame({c1: np.random.choice(col1_vals, size=num_rows),
                                            c2: np.random.randint(0, num_target_val, size=num_rows)})
                # Add two rows of the attack configuration
                df = pd.concat([df, pd.DataFrame({c1: ['z'], c2: [0]})], ignore_index=True)
                df = pd.concat([df, pd.DataFrame({c1: ['z'], c2: [0]})], ignore_index=True)
                
                # create a uniform random integer between 0 and num_target_val
                target_val = np.random.randint(0, num_target_val)
                df = pd.concat([df, pd.DataFrame({c1: ['z'], c2: [target_val]})], ignore_index=True)

                # Need to shuffle the dataframes otherwise we'll get the same
                # noise due to the same indices assigned by syndiffix
                df = df.sample(frac=1).reset_index(drop=True)

                syn = Synthesizer(df,
                    anonymization_params=AnonymizationParams(low_count_params=SuppressionParams(low_mean_gap=low_mean_gap)))
                df_syn = syn.sample()
                num_rows_with_z_and_0 = len(df_syn[(df_syn[c1] == 'z') & (df_syn[c2] == 0)])
                if num_rows_with_z_and_0 > 0:
                    # positive guess
                    if target_val == 0:
                        # correct
                        results[rm_key][ntv_key][lmg_key]['tp'] += 1
                    else:
                        # wrong
                        results[rm_key][ntv_key][lmg_key]['fp'] += 1
                        print(f"Found a false positive for {rm_key}, {ntv_key}, {lmg_key} on try {this_try}")
                        tw = TreeWalker(syn)
                        pp.pprint(tw.get_forest_nodes())
                else:
                    # negative guess
                    if target_val == 0:
                        # wrong
                        results[rm_key][ntv_key][lmg_key]['fn'] += 1
                    else:
                        # correct
                        results[rm_key][ntv_key][lmg_key]['tn'] += 1

        print(results)
# Dump results as a json file
with open('suppress_threshold_results.json', 'w') as f:
    json.dump(results, f, indent=4)