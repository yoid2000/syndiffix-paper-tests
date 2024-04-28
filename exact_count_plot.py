import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import seaborn as sns

window = 500

def apply_limits(df, limits, sort_col):
    xlabel = ','
    for col, limit in limits.items():
        if col == sort_col:
            continue
        if limit[1] is None:
            continue
        if limit[0] == 'min':
            xlabel += f' {col}>={limit[1]},'
            df = df[df[col] >= limit[1]]
        else:
            xlabel += f' {col}<={limit[1]},'
            df = df[df[col] <= limit[1]]
    xlabel = xlabel[:-1]
    return df, xlabel

def split_sort_and_roll(df, sort_col, all_col, smooth=False, limits=None):
    # make a copy of all_col
    other_col = all_col.copy()
    # remove sort_col from all_col
    other_col.remove(sort_col)
    # make dataframe df_agg_size with columns agg_size, dim, and correct 
    df_temp = df.copy()
    df_temp = df_temp[[sort_col, 'dim', 'correct']+other_col]
    min = df_temp[sort_col].min()
    # Assign a random float to each row in the sort_col column between the sort_col value and sort_col value + 1
    if smooth:
        df_temp[sort_col] = df_temp[sort_col].astype(float)
        for i in df_temp.index:
            # We don't randomize the minimum value so that the rolling average
            # shows the minimum value in the plot
            if df_temp.loc[i, sort_col] != min:
                df_temp.loc[i, sort_col] = df_temp.loc[i, sort_col] + np.random.rand()
    df_list = [None, None, None, None]
    if limits is None:
        xlabel = ''
    else:
        df_temp, xlabel = apply_limits(df_temp, limits, sort_col)
    # make a dataframe per attack dimension
    for dim in [1, 2, 3]:
        df_list[dim] = df_temp[df_temp['dim'] == dim]
        # sort agg_size
        df_list[dim] = df_list[dim].sort_values(sort_col)
        # add two new columns, precision and roll, where precision is the rolling average of correct with a window size of window and roll is the rolling average of agg_size with a window size of window
        df_list[dim]['precision'] = df_list[dim]['correct'].rolling(window=window).mean()
        df_list[dim]['roll'] = df_list[dim][sort_col].rolling(window=window).mean()
        # drop rows with NaN values
        df_list[dim] = df_list[dim].dropna()
    # concatinate the three dataframes into one
    df_concat = pd.concat([df_list[1], df_list[2], df_list[3]])
    # shuffle df_concat
    df_concat = df_concat.sample(frac=1)
    # sort df_concat by roll
    df_concat = df_concat.sort_values('roll')
    return df_concat, xlabel

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
all_cols = ['agg_size', 'num_val', 'num_col', 'frac_leaf_under']

# Define the number of bins for each column
num_bins = 6

# Define the combinations of the three variables
combinations = [('agg_size', 'num_val', 'num_col')]

# For each 'dim' value, create a plot
for dim in [1, 2, 3]:
    # Filter the DataFrame for the current 'dim' value
    df_dim = df[df['dim'] == dim]

    # Create bins for 'agg_size', 'num_val', and 'num_col'
    for var in ['agg_size', 'num_val', 'num_col']:
        df_dim[var + '_bin'] = pd.cut(df_dim[var], bins=num_bins)

    # Create a figure with 6 subplots
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    axs = axs.flatten()

    # For each 'agg_size' bin, create a heatmap
    for i, (agg_size_bin, df_agg_size_bin) in enumerate(df_dim.groupby('agg_size_bin')):
        # Calculate precision for each bin
        precision_per_bin = df_agg_size_bin.groupby(['num_val_bin', 'num_col_bin'])['correct'].mean().reset_index()

        # Pivot the data for the heatmap
        precision_pivot = precision_per_bin.pivot(index='num_val_bin', columns='num_col_bin', values='correct')

        # Create the heatmap
        sns.heatmap(precision_pivot, cmap='YlGnBu', linewidths=0.5, ax=axs[i], vmin=0, vmax=0.8, annot=True, fmt=".2f")
        axs[i].set_title(f'agg_size: {agg_size_bin}')
        if i < 3:  # If it's a top heatmap, remove the X labels
            axs[i].set_xlabel('')
            axs[i].set_xticklabels([])
        if i % 3 != 0:  # If it's not a leftmost heatmap, remove the Y labels
            axs[i].set_ylabel('')
            axs[i].set_yticklabels([])

    # Add a colorbar
    #fig.colorbar(axs[0].collections[0], ax=axs, location="right", use_gridspec=False, pad=0.2)

    # Show the plot
    plt.tight_layout()
    file_path = os.path.join('results', f'exact_count_3way.dim{dim}.png')
    plt.savefig(file_path, bbox_inches='tight', dpi=300)	
    plt.close()

# Define the combinations of the three variables
combinations = [('agg_size', 'num_val'), ('agg_size', 'num_col'), ('num_val', 'num_col')]

# For each combination, create a heatmap
for dim in [1, 2, 3]:
    df_dim = df.copy()
    df_dim = df_dim[df_dim['dim'] == dim]
    for i, (var1, var2) in enumerate(combinations):
        # Create bins for var1 and var2
        df_dim[var1 + '_bin'] = pd.qcut(df_dim[var1], 10, duplicates='drop')
        df_dim[var2 + '_bin'] = pd.qcut(df_dim[var2], 10, duplicates='drop')

        # Calculate precision for each bin
        precision_per_bin = df_dim.groupby([var1 + '_bin', var2 + '_bin'])['correct'].mean().reset_index()

        # Convert the interval categories to strings
        #precision_per_bin[var1 + '_bin'] = precision_per_bin[var1 + '_bin'].astype(str)
        #precision_per_bin[var2 + '_bin'] = precision_per_bin[var2 + '_bin'].astype(str)

        # Pivot the data for the heatmap
        precision_pivot = precision_per_bin.pivot(index=var1 + '_bin', columns=var2 + '_bin', values='correct')

        # Create the heatmap
        plt.figure(i)
        sns.heatmap(precision_pivot, cmap='YlGnBu', linewidths=0.5)
        plt.title(f'Precision for {var1} and {var2}, dim {dim}')
        plt.xlabel(var2)
        plt.ylabel(var1)
        plt.tight_layout()
        file_path = os.path.join('results', f'exact_count_heatmap.dim{dim}.{var1}.{var2}.png')
        plt.savefig(file_path, bbox_inches='tight', dpi=300)	
        plt.close()

def make_line_plots(df, limits=None, tag=None):
    # Create a line plot for each variable
    df_agg_size, agg_size_label = split_sort_and_roll(df, 'agg_size', all_cols, smooth=True, limits=limits)
    df_num_val, num_val_label = split_sort_and_roll(df, 'num_val', all_cols, smooth=True, limits=limits)
    df_num_col, num_col_label = split_sort_and_roll(df, 'num_col', all_cols, smooth=True, limits=limits)
    df_frac_leaf_under, _ = split_sort_and_roll(df, 'frac_leaf_under', all_cols)

    # Define the color map for 'dim'
    color_map = {1: 'red', 2: 'green', 3: 'blue'}

    # Calculate the global minimum and maximum precision values
    min_precision = min(df['precision'].min() for df in [df_agg_size, df_num_val, df_num_col, df_frac_leaf_under])
    max_precision = max(df['precision'].max() for df in [df_agg_size, df_num_val, df_num_col, df_frac_leaf_under])

    print(f"max precision {max_precision} for rolling average window {window}")

    # Create a 2x2 subplot
    fig, axs = plt.subplots(2, 2, figsize=(6, 6))

    # Create line plots for df_agg_size, df_num_val, and df_num_col
    for df, ax, xlabel in zip([df_agg_size, df_num_val, df_num_col], axs.flatten()[:3], ['Target aggregate size'+agg_size_label, 'Distinct values'+num_val_label, 'Columns'+num_col_label]):
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
    axs[1, 1].set_xlabel('Fraction suppressed nodes')
    axs[1, 1].set_ylabel('Precision')
    axs[1, 1].set_ylim([min_precision, max_precision])  # Set the y-axis range

    # Create a legend for 'dim'
    legend_elements = [mlines.Line2D([], [], color=color, label=f"Attack dims {dim}") for dim, color in color_map.items()]
    fig.legend(handles=legend_elements, loc='upper center', ncol=3)

    plt.tight_layout()

    if tag is None:
        file_name = f'exact_count.win{window}.png'
    else:
        file_name = f'exact_count.{tag}.win{window}.png'
    file_path = os.path.join('results', file_name)
    plt.savefig(file_path, bbox_inches='tight', dpi=300)	

    plt.close()

limits = {'agg_size': ['min', 100], 'num_val': ['max', 10], 'num_col': ['min', None]}	
make_line_plots(df, limits=limits, tag='lim1')
limits = {'agg_size': ['min', 100], 'num_val': ['max', 5], 'num_col': ['min', None]}	
make_line_plots(df, limits=limits, tag='lim2')
make_line_plots(df)