import sys
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

num_rows = 300
num_cols = [20, 40, 80, 160]
# We want more runs for lower dimension data because each run has fewer samples
# We need more runs to get significant results (3-column measures take a long time
# to run)
min_samples = 10000
TWO_COLS = True
THREE_COLS = True
attack_keys = ['1dim', '2dim', '3dim']

# read command line arguments
attack = 'do_simple'
if len(sys.argv) > 1:
    attack = sys.argv[1]

if attack == 'do_simple':
    outFile = 'exact_count_precision_simple.json'	
elif attack == 'do_least_accurate':
    outFile = 'exact_count_precision_least_accurate.json'
elif attack == 'do_random':
    outFile = 'exact_count_precision_random.json'

def print_progress_wheel(wheel):
    print(next(wheel) + '\b', end='', flush=True)

def progress_wheel():
    wheel = itertools.cycle(['-', '/', '|', '\\'])
    while True:
        yield next(wheel)

def get_precision(noisy_counts, exact_counts):
    num_correct = 0
    # Assume that we can determine the exact number of rows by taking
    # average from all tables
    true_row_count = sum(exact_counts)
    # We know that the sum of noisy counts should be equal to the true row 
    # count, so if not we adjust to make it so
    guessed_floats = [sum(noisy_counts[i])/len(noisy_counts[i]) for i in [0,1]]
    guessed_rounded = [round(g) for g in guessed_floats]
    if attack == 'do_least_accurate':
        # The the decimal part of the guessed_floats
        guessed_decimals = [g - int(g) for g in guessed_floats]
        # These guessed_counts are floating point numbers. Although it
        # probably doesn't much matter, let's assume that if we need to 
        # adjust one count, it is best to adjust the one that is the
        # least accurate.
        guessed_acc = [abs(0.5 - g) for g in guessed_decimals]
        if guessed_acc[0] < guessed_acc[1]:
            adjust_target = 0
        else:
            adjust_target = 1
    elif attack == 'do_random':
        # Randomly choose which count to adjust
        adjust_target = np.random.randint(0,2)
    if attack != 'do_simple':
        adjustment = true_row_count - sum(guessed_rounded)
        while abs(adjustment) > 1:
            # If we need to adjust by two, then we adjust both counts
            if adjustment > 0:
                for i in [0,1]:
                    guessed_rounded[i] += 1
                adjustment -= 2
            else:
                for i in [0,1]:
                    guessed_rounded[i] -= 1
                adjustment += 2
        # Now we need to adjust by -1, 0, or 1
        if adjustment == 1:
            guessed_rounded[adjust_target] += 1
        elif adjustment == -1:
            guessed_rounded[adjust_target] -= 1
    guesses_correct = []
    for i in [0,1]:
        guess = guessed_rounded[i]
        if guess == exact_counts[i]:
            guesses_correct.append(1)
        else:
            guesses_correct.append(0)
    errors = [abs(guessed_rounded[i] - exact_counts[i]) for i in [0,1]]
    return {'correct': guesses_correct, 'guessed': guessed_rounded, 'exact': exact_counts, 'errors': errors}

def summarize_and_dump(precision, ckey):
    precision[ckey]['correct_averages']['1dim'] = statistics.mean(precision[ckey]['scores']['1dim'])
    precision[ckey]['error_averages']['1dim'] = statistics.mean(precision[ckey]['errors']['1dim'])
    precision[ckey]['error_std_devs']['1dim'] = statistics.stdev(precision[ckey]['errors']['1dim'])
    precision[ckey]['samples']['1dim'] = len(precision[ckey]['scores']['1dim'])
    if TWO_COLS:
        precision[ckey]['correct_averages']['2dim'] = statistics.mean(precision[ckey]['scores']['2dim'])
        precision[ckey]['error_averages']['2dim'] = statistics.mean(precision[ckey]['errors']['1dim'])
        precision[ckey]['error_std_devs']['2dim'] = statistics.stdev(precision[ckey]['errors']['2dim'])
        precision[ckey]['samples']['2dim'] = len(precision[ckey]['scores']['2dim'])
    if THREE_COLS:
        precision[ckey]['correct_averages']['3dim'] = statistics.mean(precision[ckey]['scores']['3dim'])
        precision[ckey]['error_averages']['3dim'] = statistics.mean(precision[ckey]['errors']['1dim'])
        precision[ckey]['error_std_devs']['3dim'] = statistics.stdev(precision[ckey]['errors']['3dim'])
        precision[ckey]['samples']['3dim'] = len(precision[ckey]['scores']['3dim'])
    print(precision)
    # dump precision as a json file
    with open(outFile, 'w') as f:
        json.dump(precision, f, indent=4)

wheel = progress_wheel()
precision = {}
# get both the index and the value for the list num_cols

for cix, c in enumerate(num_cols):
    ckey = f"{c} cols"
    precision[ckey] = {
                    'correct_averages': {'1dim': 0, '2dim': 0, '3dim': 0},
                    'error_averages': {'1dim': 0, '2dim': 0, '3dim': 0},
                    'error_std_devs': {'1dim': 0, '2dim': 0, '3dim': 0},
                    'samples': {'1dim': 0, '2dim': 0, '3dim': 0},
                    'scores': {'1dim': [], '2dim': [], '3dim': []},
                    'results': {'1dim': [], '2dim': [], '3dim': []},
                    }
    num_correct = [0,0]
    noisy_counts = [[[],[]], [[],[]], [[],[]]]
    samples_per_2col = max(20, min_samples / (c - 1))
    samples_per_3col = max(20, min_samples / (((c-1) * (c-2)) / 2))
    for this_try in range(min_samples):
        df = pd.DataFrame(np.random.randint(0, 2, size=(num_rows, c)), 
                          columns=[f'col{this_try}_{i}' for i in range(c)])
        true_row_count = df.shape[0]
        col0 = f'col{this_try}_0'
        cols_without_col0 = [col for col in df.columns if col != col0]
        # get the count of each value for col0 in df
        exact_counts = [0,0]
        for i in [0,1]:
            exact_counts[i] = df[df[col0] == i].shape[0]
        df_syn = Synthesizer(df[[col0]]).sample()
        for i in [0,1]:
            noisy_counts[0][i].append(df_syn[df_syn[col0] == i].shape[0])
        results = get_precision(noisy_counts[0], exact_counts)
        precision[ckey]['results']['1dim'].append(results)
        precision[ckey]['scores']['1dim'] += results['correct']
        precision[ckey]['errors']['1dim'] += results['errors']
        if TWO_COLS and this_try <= samples_per_2col:
            for col in cols_without_col0:
                df_syn = Synthesizer(df[[col0,col]]).sample()
                #print_progress_wheel(wheel)
                for i in [0,1]:
                    noisy_counts[1][i].append(df_syn[df_syn[col0] == i].shape[0])
            results = get_precision(noisy_counts[1], exact_counts)
            precision[ckey]['results']['2dim'].append(results)
            precision[ckey]['scores']['2dim'] += results['correct']
            precision[ckey]['errors']['2dim'] += results['errors']
            #print(f"{c}-{this_try}.2 (of {samples_per_2col})", flush=True)
        if THREE_COLS and this_try <= samples_per_3col:
            for comb in itertools.combinations(cols_without_col0, 2):
                cols = [col0] + list(comb)
                df_syn = Synthesizer(df[cols]).sample()
                #print_progress_wheel(wheel)
                for i in [0,1]:
                    noisy_counts[2][i].append(df_syn[df_syn[col0] == i].shape[0])
            results = get_precision(noisy_counts[2], exact_counts)
            precision[ckey]['results']['3dim'].append(results)
            precision[ckey]['scores']['3dim'] += results['correct']
            precision[ckey]['errors']['2dim'] += results['errors']
            print(f"{c}-{this_try}.3 (of {samples_per_3col})", flush=True)
            summarize_and_dump(precision, ckey)
    summarize_and_dump(precision, ckey)