import pandas as pd
import numpy as np
from syndiffix import Synthesizer
import itertools
import statistics
from collections import Counter
from tabulate import tabulate
import pprint

def build_random_dataframe():
    # Define the number of rows and the range of integers
    num_rows = 50000
    num_integers = 10
    
    # Generate random integers for each column
    col1 = np.random.randint(0, num_integers, size=num_rows)
    col2 = np.random.randint(0, num_integers, size=num_rows)
    col3 = np.random.randint(0, num_integers, size=num_rows)
    
    # Create the DataFrame
    df = pd.DataFrame({
        'col1': col1,
        'col2': col2,
        'col3': col3
    })
    
    return df

def generate_column_combinations(columns):
    all_combinations = []
    for r in range(1, len(columns) + 1):
        combinations = list(itertools.combinations(columns, r))
        all_combinations.extend(combinations)
    return all_combinations

def get_min_count(df):
    # Get the list of column names
    columns = df.columns.tolist()
    
    min_count = float('inf')
    
    # Iterate over all combinations of 1, 2, and 3 columns
    for r in range(1, len(columns) + 1):
        for combination in itertools.combinations(columns, r):
            # Group by the combination of columns and get the size of each group
            group_sizes = df.groupby(list(combination)).size()
            
            # Get the minimum count in the current group
            current_min_count = group_sizes.min()
            
            # Update the overall minimum count
            if current_min_count < min_count:
                min_count = current_min_count
    
    return min_count

import itertools
import pandas as pd

def count_differences(df, df_syn):
    # Get the list of column names
    columns = df.columns.tolist()
    
    count_diff_dict = {}
    
    # Use combinations of all three columns
    combination = columns
    
    # Group by the combination of columns and get the size of each group for both dataframes
    df_group_sizes = df.groupby(combination).size()
    df_syn_group_sizes = df_syn.groupby(combination).size()
    
    # Get the union of all unique combinations in both dataframes
    all_combinations = set(df_group_sizes.index).union(set(df_syn_group_sizes.index))
    
    # Calculate the count difference for each combination
    for comb in all_combinations:
        count_df = df_group_sizes.get(comb, 0)
        count_df_syn = df_syn_group_sizes.get(comb, 0)
        count_diff_dict[comb] = count_df - count_df_syn
    
    return count_diff_dict


def calculate_statistics(count_diff_dict):
    values = list(count_diff_dict.values())
    
    mean_val = statistics.mean(values)
    stddev_val = statistics.stdev(values)
    min_val = min(values)
    max_val = max(values)
    
    return mean_val, stddev_val, min_val, max_val


def list_value_counts(count_diff_dict):
    # Count the occurrences of each value in count_diff_dict
    value_counts = Counter(count_diff_dict.values())
    
    # Sort the value counts by the value (key)
    sorted_value_counts = sorted(value_counts.items())
    
    return sorted_value_counts


def print_table(sorted_value_counts):
    # Create a table with headers
    table = [["Error", "Count"]]
    table.extend(sorted_value_counts)
    
    # Print the table using tabulate
    print(tabulate(table, headers="firstrow", tablefmt="grid"))


pp = pprint.PrettyPrinter(indent=4)

df = build_random_dataframe()

min_count = get_min_count(df)
print(f'The minimum count is: {min_count}')

df_syn = Synthesizer(df).sample()

count_diff_dict = count_differences(df, df_syn)
pp.pprint(count_diff_dict)

mean_val, stddev_val, min_val, max_val = calculate_statistics(count_diff_dict)

print(f'Mean: {mean_val}')
print(f'Standard Deviation: {stddev_val}')
print(f'Minimum: {min_val}')
print(f'Maximum: {max_val}')
sorted_value_counts = list_value_counts(count_diff_dict)
print_table(sorted_value_counts)