import json
import pprint
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os


pp = pprint.PrettyPrinter(indent=4)

# read file suppress_threshold_results.json
with open('suppress_threshold_results.json', 'r') as f:
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
            rows_ratio = f"{((gapv['num_rows'] / tar) / 3):.1f}"
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
            else:
                precision = gapv['tp'] / all_pos
            precision_improvement = (precision - stat_guess) / (1.0 - stat_guess) 
            coverage = all_pos / gapv['samples']
            # A value of 0 would screw up the log scale
            coverage = max(1/30000, coverage)
            data.append({
                'sd_gap': str(int(gap)),
                'mult': str(mult),
                'num_targets': str(tar),
                'mean': mean,
                'precision': precision,
                'precision_improvement': precision_improvement,
                'coverage': coverage,
                'rows_ratio': str(int(rows_ratio)),
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
print(df[['sd_gap','mult','num_targets','num_rows']].to_string())

# Create a color map for the 'rows_ratio' column
rows_ratio_colors = {rows_ratio_value: color for rows_ratio_value, color in zip(df['rows_ratio'].unique(), ['orange', 'red', 'blue'])}
df['color'] = df['rows_ratio'].map(rows_ratio_colors)

# Create a marker map for the 'mean' column
mean_markers = {mean_value: marker for mean_value, marker in zip(df['mean'].unique(), ['o', 's', '^'])}
df['marker'] = df['mean'].map(mean_markers)

# Create the scatter plot
plt.figure(figsize=(9, 4.5))
for marker, mean in zip(df['marker'].unique(), df['mean'].unique()):
    df_marker = df[df['marker'] == marker]
    for color, rows_ratio in zip(df_marker['color'].unique(), df_marker['rows_ratio'].unique()):
        df_marker_color = df_marker[df_marker['color'] == color]
        plt.scatter(df_marker_color['coverage'], df_marker_color['precision_improvement'], c=df_marker_color['color'], marker=marker, label=f'rows_ratio={rows_ratio}, mean={mean}')

plt.xscale('log')
plt.xlabel('Coverage (log scale)', fontsize=13, labelpad=10)
plt.ylabel('Precision Improvement', fontsize=13, labelpad=10)

# Add horizontal lines
plt.axhline(0.0, color='black', linestyle='--')
plt.axhline(0.5, color='black', linestyle='--')

# Modify x-axis ticks and labels
ticks = list(plt.xticks()[0]) + [1/30000]
labels = [t if t != 1/30000 else 'None' for t in ticks]
plt.xticks(ticks, labels)

# Set x-axis range to min and max 'coverage' values
plt.xlim(1/(30000 + 5000), 0.02)

plt.legend(ncol=2)

# make a path to suppress.png in the results directory
# Create the path to the results directory
os.makedirs('results', exist_ok=True)

# Create the path to suppress.png
path_to_suppress_png = os.path.join('results', 'suppress.png')

# Save the plot as a PNG file
plt.savefig(path_to_suppress_png, dpi=300, bbox_inches='tight')

plt.close()