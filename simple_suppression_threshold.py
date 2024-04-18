import pandas as pd
import numpy as np
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

low_mean_gaps = [2.0, 3.0, 4.0]
num_tries = 1000

results = {}
for low_mean_gap in low_mean_gaps:
    lmg_key = f'{low_mean_gap} low_mean_gap'
    results[lmg_key] = {'tp':0, 'fp':0, 'tn':0, 'fn':0}
    col1_vals = ['a', 'b', 'c']
    # Compute num_rows such that there are not many suppressed combinations
    num_rows = len(col1_vals) * 10
    for this_try in range(num_tries):
        # Use different column names with each run so as to get different noise
        c1 = f"c1_{this_try}"
        df_pos = pd.DataFrame({c1: np.random.choice(col1_vals, size=num_rows)})
        # Add two rows of the attack configuration
        df_pos = pd.concat([df_pos, pd.DataFrame({c1: ['z']})], ignore_index=True)
        df_pos = pd.concat([df_pos, pd.DataFrame({c1: ['z']})], ignore_index=True)

        # Make an exact copy of df_pos
        df_neg = df_pos.copy()
        # Add the victim, with the target value (positive guess)
        df_pos = pd.concat([df_pos, pd.DataFrame({c1: ['z']})], ignore_index=True)
        # Add the victim with a different target value (negative guess)
        df_neg = pd.concat([df_neg, pd.DataFrame({c1: ['a']})], ignore_index=True)

        df_pos = df_pos.sample(frac=1).reset_index(drop=True)
        df_neg = df_neg.sample(frac=1).reset_index(drop=True)

        df_syn_pos = Synthesizer(df_pos,
            anonymization_params=AnonymizationParams(low_count_params=SuppressionParams(low_mean_gap=low_mean_gap, layer_sd=1.0, low_threshold=3))).sample()
        num_rows_with_z_and_0_pos = len(df_syn_pos[(df_syn_pos[c1] == 'z')])
        if num_rows_with_z_and_0_pos > 0:
            # positive guess, which is correct
            results[lmg_key]['tp'] += 1
        else:
            # negative guess, which is incorrect
            results[lmg_key]['fn'] += 1

        df_syn_neg = Synthesizer(df_neg,
            anonymization_params=AnonymizationParams(low_count_params=SuppressionParams(low_mean_gap=low_mean_gap, layer_sd=1.0, low_threshold=3))).sample()
        num_rows_with_z_and_0_neg = len(df_syn_neg[(df_syn_neg[c1] == 'z')])
        if num_rows_with_z_and_0_neg > 0:
            # positive guess, which is incorrect
            results[lmg_key]['fp'] += 1
        else:
            # negative guess, which is correct
            results[lmg_key]['tn'] += 1

        #print(df_syn_pos.to_string())
        #print(df_syn_neg.to_string())
        #quit()
    print(results)
# Dump results as a json file
with open('suppress_threshold_results.json', 'w') as f:
    json.dump(results, f, indent=4)