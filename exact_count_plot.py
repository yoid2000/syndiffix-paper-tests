import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import seaborn as sns


window = 200

def split_sort_and_roll(df, col, smooth=False):
    # make dataframe df_agg_size with columns agg_size, dim, and correct 
    df_temp = df.copy()
    df_temp = df_temp[[col, 'dim', 'correct']]
    # Assign a random float to each row in the col column between the col value and col value + 1
    if smooth:
        df_temp[col] = df_temp[col].astype(float)
        for i in df_temp.index:
            df_temp.loc[i, col] = df_temp.loc[i, col] + np.random.rand()
    # make a dataframe per attack dimension
    df_1 = df_temp[df_temp['dim'] == 1]
    df_2 = df_temp[df_temp['dim'] == 2]
    df_3 = df_temp[df_temp['dim'] == 3]
    # sort agg_size
    df_1 = df_1.sort_values(col)
    df_2 = df_2.sort_values(col)
    df_3 = df_3.sort_values(col)
    # add two new columns, precision and roll, where precision is the rolling average of correct with a window size of window and roll is the rolling average of agg_size with a window size of window
    df_1['precision'] = df_1['correct'].rolling(window=window).mean()
    df_1['roll'] = df_1[col].rolling(window=window).mean()
    df_2['precision'] = df_2['correct'].rolling(window=window).mean()
    df_2['roll'] = df_2[col].rolling(window=window).mean()
    df_3['precision'] = df_3['correct'].rolling(window=window).mean()
    df_3['roll'] = df_3[col].rolling(window=window).mean()
    # drop rows with NaN values
    df_1 = df_1.dropna()
    df_2 = df_2.dropna()
    df_3 = df_3.dropna()
    # concatinate the three dataframes into one
    df_concat = pd.concat([df_1, df_2, df_3])
    # shuffle df_concat
    df_concat = df_concat.sample(frac=1)
    # sort df_concat by roll
    df_concat = df_concat.sort_values('roll')
    return df_concat

json_files = [pos_json for pos_json in os.listdir("./exact_count_results") if pos_json.endswith('.json')]

data = []

for file in json_files:
    with open(f"./exact_count_results/{file}", "r") as json_file:
        print(file)
        dat1 = json.load(json_file)
        print(f"read {len(dat1)} records")
        for thing in dat1:
            data.append(thing)

print(f"total records {len(data)}")

df = pd.DataFrame.from_records(data)
# make a new column which is 1 - frac_leaf_over
df['frac_leaf_under'] = 1 - df['frac_leaf_over']

df_agg_size = split_sort_and_roll(df, 'agg_size', smooth=True)
df_num_val = split_sort_and_roll(df, 'num_val', smooth=True)
df_num_col = split_sort_and_roll(df, 'num_col', smooth=True)
df_frac_leaf_under = split_sort_and_roll(df, 'frac_leaf_under')

# Define the color map for 'dim'
color_map = {1: 'red', 2: 'green', 3: 'blue'}

# Calculate the global minimum and maximum precision values
min_precision = min(df['precision'].min() for df in [df_agg_size, df_num_val, df_num_col, df_frac_leaf_under])
max_precision = max(df['precision'].max() for df in [df_agg_size, df_num_val, df_num_col, df_frac_leaf_under])

print(f"max precision {max_precision} for rolling average window {window}")

# Create a 2x2 subplot
fig, axs = plt.subplots(2, 2, figsize=(10, 10))

# Create line plots for df_agg_size, df_num_val, and df_num_col
for df, ax, xlabel in zip([df_agg_size, df_num_val, df_num_col], axs.flatten()[:3], ['Target aggregate size', 'Distinct column values', 'Columns']):
    for dim, color in color_map.items():
        df_dim = df[df['dim'] == dim]
        ax.plot(df_dim['roll'], df_dim['precision'], color=color)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Precision')
    ax.set_ylim([min_precision, max_precision])  # Set the y-axis range

# Create a scatter plot for df_frac_leaf_under
for dim, color in color_map.items():
    df_dim = df_frac_leaf_under[df_frac_leaf_under['dim'] == dim]
    axs[1, 1].scatter(df_dim['roll'], df_dim['precision'], color=color, s=5)
axs[1, 1].set_xlabel('Suppressed fraction')
axs[1, 1].set_ylabel('Precision')
axs[1, 1].set_ylim([min_precision, max_precision])  # Set the y-axis range

# Create a legend for 'dim'
legend_elements = [mlines.Line2D([], [], color=color, label=f"Attack dims {dim}") for dim, color in color_map.items()]
fig.legend(handles=legend_elements, loc='upper center', ncol=3)

plt.tight_layout()

file_path = os.path.join('results', 'exact_count.png')
plt.savefig(file_path, bbox_inches='tight', dpi=300)	

plt.close()