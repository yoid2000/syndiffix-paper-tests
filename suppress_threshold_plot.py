import json
import pprint
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
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
            rows_ratio = (gapv['num_rows'] / tar) / 3
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
                'sd_gap': int(gap),
                'mult': int(mult),
                'num_targets': int(tar),
                'mean': mean,
                'precision': precision,
                'precision_improvement': precision_improvement,
                'coverage': coverage,
                'rows_ratio': int(rows_ratio),
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
print(df[['sd_gap','sd_gap','num_targets','rows_ratio']].to_string())

# Define the marker styles for sd_gap
marker_map = {2: 'o', 3: '^', 4: 'v'}

# Define the color map for rows_ratio
colors = sns.color_palette()[:3]
color_map = {5: colors[0], 10: colors[1], 100: colors[2]}

# Define the size map for num_targets
size_map = {2: 50, 5: 100, 10: 150}

# Create the scatter plot
plt.figure(figsize=(8, 4))

for (sd_gap, marker) in marker_map.items():
    for (rows_ratio, color) in color_map.items():
        for (num_targets, size) in size_map.items():
            df_filtered = df[(df['sd_gap'] == sd_gap) & (df['rows_ratio'] == rows_ratio) & (df['num_targets'] == num_targets)]
            print(sd_gap, rows_ratio, num_targets)
            print(df_filtered.to_string())
            plt.scatter(df_filtered['coverage'], df_filtered['precision_improvement'], color=color, marker=marker, s=size, alpha=0.8)

# Add horizontal lines
plt.axhline(0, color='black', linestyle='--')
plt.axhline(0.5, color='black', linestyle='--')

# Set axis labels
plt.xscale('log')
plt.xlabel('Coverage (log scale)', fontsize=13, labelpad=10)
plt.ylabel('Precision Improvement', fontsize=13, labelpad=10)

# Create legends
legend1 = plt.legend([mlines.Line2D([0], [0], color='black', marker=marker, linestyle='None') for sd_gap, marker in marker_map.items()], ['sd_gap: {}'.format(sd_gap) for sd_gap in marker_map.keys()], title='', loc='lower left', bbox_to_anchor=(0.3, 0), fontsize='small')
legend2 = plt.legend([mlines.Line2D([0], [0], color=color, marker='o', linestyle='None') for rows_ratio, color in color_map.items()], ['rows_ratio: {}'.format(rows_ratio) for rows_ratio in color_map.keys()], title='', loc='lower left', bbox_to_anchor=(0.5, 0), fontsize='small')
legend3 = plt.legend([mlines.Line2D([0], [0], color='black', marker='o', markersize=size/10, linestyle='None') for num_targets, size in size_map.items()], ['num_targets: {}'.format(num_targets) for num_targets in size_map.keys()], title='', loc='lower left', bbox_to_anchor=(0.75, 0), fontsize='small')

plt.gca().add_artist(legend1)
plt.gca().add_artist(legend2)

# Modify x-axis ticks and labels
ticks = list(plt.xticks()[0]) + [1/30000]
labels = [t if t != 1/30000 else 'None' for t in ticks]
plt.xticks(ticks, labels)

# Set x-axis range to min and max 'coverage' values
plt.xlim(1/(30000 + 5000), 0.02)

# Adjust the layout
plt.subplots_adjust(right=0.85)  # Adjust this value as needed
plt.tight_layout()

plt.show()

# Create the path to suppress.png
path_to_suppress_png = os.path.join('results', 'suppress.png')

# Save the plot as a PNG file
plt.savefig(path_to_suppress_png, dpi=300, bbox_inches='tight')

plt.close()