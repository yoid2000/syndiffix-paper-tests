import argparse
import os
import pandas as pd
import numpy as np
import json
import sys
import random
from syndiffix import Synthesizer
import alscore
import pprint
import sys

'''
All outlier tests are based on detecting whether there are:
        0 or 1 outlier
        1 or 2 outliers
        2 or 3 outliers
Value outlier tests:
    Detect presence:
        We know the presence of the outlier will raise the average value
        of the column with the outlier value. But, we need to know what the expected value without the outlier should be...
    Infer value of another column:
'''

num_1col_runs = 500000
prediction_multipliers = [1, 2, 3]
if 'SDX_TEST_DIR' in os.environ:
    base_path = os.getenv('SDX_TEST_DIR')
else:
    base_path = os.getcwd()
if 'SDX_TEST_CODE' in os.environ:
    code_path = os.getenv('SDX_TEST_CODE')
else:
    code_path = None
os.makedirs(base_path, exist_ok=True)
runs_path = os.path.join(base_path, 'outlier_theory')
os.makedirs(runs_path, exist_ok=True)
tests_path = os.path.join(runs_path, 'tests')
os.makedirs(tests_path, exist_ok=True)
results_path = os.path.join(runs_path, 'results')
os.makedirs(results_path, exist_ok=True)
pp = pprint.PrettyPrinter(indent=4)


def build_basic_table(num_vals, ex_factor, num_ex, dist, num_aid):
    # Generate a random 5-digit number for the column names
    random_suffix = ''.join(random.choices('0123456789', k=5))
    aid_col = f'aid_{random_suffix}'
    vals_col = f'vals_{random_suffix}'

    # Generate num_aid distinct aid values
    aid_values = random.sample(range(10000000), num_aid)
    aid_values = [str(e) for e in aid_values]
    
    # Select the unknown_aid
    unknown_aid = random.choice(aid_values)
    
    # Select the known_aid values
    known_aid_values = random.sample([aid for aid in aid_values if aid != unknown_aid], num_ex - 1)
    
    # Select the other_aid values
    other_aid_values = [aid for aid in aid_values if aid != unknown_aid and aid not in known_aid_values]
    
    # Generate num_vals distinct values for the vals column
    vals_values = random.sample(range(1000, 100000), num_vals)
    vals_values = [str(e) for e in vals_values]
    print(f'vals_values: {vals_values}')
    if len(vals_values) != num_vals:
        print(f"Fail: Expected {num_vals} distinct values, but found {len(vals_values)}.")
        print(vals_values)
        sys.exit(1)
    
    # Assign a random val to each aid
    aid_to_val = {aid: random.choice(vals_values) for aid in aid_values}
    
    # Ensure the known_aid has the same val, known_val
    known_val = random.choice(vals_values)
    for aid in known_aid_values:
        aid_to_val[aid] = known_val
    
    # Create the DataFrame
    rows = []
    
    # Add rows for other_aid values
    max_rows = 0
    for aid in other_aid_values:
        if dist == 'uniform':
            num_rows = random.randint(40, 60)
        elif dist == 'normal':
            num_rows = max(40, int(random.normalvariate(80, 20)))
        if num_rows > max_rows:
            max_rows = num_rows
        for _ in range(num_rows):
            rows.append({aid_col: aid, vals_col: aid_to_val[aid]})
    
    # Add rows for known_aid values
    for aid in known_aid_values:
        for _ in range(max_rows * ex_factor):
            rows.append({aid_col: aid, vals_col: known_val})
    
    # Add rows for unknown_aid
    unknown_val = aid_to_val[unknown_aid]
    for _ in range(15 * ex_factor):
        rows.append({aid_col: unknown_aid, vals_col: unknown_val})
    
    df = pd.DataFrame(rows)
    
    # Shuffle the DataFrame
    df = df.sample(frac=1).reset_index(drop=True)
    df = df.astype(str)

    
    return df, known_val, unknown_aid, unknown_val

def make_slurm():
    pass

def make_plot():
    a1c = Attack1colResults(results_path)
    a1c.make_predictions()
    a1c.analyze()

def membership_attack():
    pass

import pandas as pd

def most_frequent_value(df):
    # Get the column name
    column_name = df.columns[0]
    
    # Use value_counts to count occurrences of each value
    value_counts = df[column_name].value_counts()
    
    # Select the most frequent value
    most_frequent = value_counts.idxmax()
    
    # Calculate gap_1_2
    if len(value_counts) > 1:
        second_most_frequent_count = value_counts.iloc[1]
    else:
        second_most_frequent_count = 0
    gap_1_2 = value_counts.iloc[0] - second_most_frequent_count
    
    # Calculate gap_avg
    if len(value_counts) > 1:
        avg_count = value_counts.iloc[1:].mean()
    else:
        avg_count = 0
    gap_avg = value_counts.iloc[0] - avg_count
    
    return most_frequent, gap_1_2, gap_avg

def run_one_1col_attack(a1c, num_vals, ex_factor, num_ex, dist, num_aid):
    # Returns 1 if predictions is correct, 0 otherwise
    df, known_val, unknown_aid, unknown_val = build_basic_table(num_vals, ex_factor, num_ex, dist, num_aid)
    # Get the value belonging to the unknown_aid
    # split df into two dataframes, one with the aid column and one with the vals column
    aid_col = [col for col in df.columns if col.startswith('aid_')][0]
    vals_col = [col for col in df.columns if col.startswith('vals_')][0]
    df_aid = df[[aid_col]]
    df_vals = df[[vals_col]]
    syn = Synthesizer(df_vals, pids=df_aid)
    df_syn = syn.sample()
    print(f"There are {df[vals_col].nunique()} uniques in df, and {df_syn[vals_col].nunique()} uniques in df_syn")
    # get the counts of all values in vals_col of df_syn
    value_counts = df_syn[vals_col].value_counts()
    value_counts_dict = value_counts.to_dict()
    value_counts_dict = {str(k): v for k, v in value_counts.items()}
    predict_val, gap_1_2, gap_avg = most_frequent_value(df_syn)
    print(f"Predict {predict_val} for unknown_val {unknown_val}")
    a1c.add(num_vals, ex_factor, num_ex, dist, num_aid, predict_val, known_val, unknown_val, gap_1_2, gap_avg, value_counts_dict)
    if predict_val == unknown_val:
        return 1, gap_1_2, gap_avg
    else:
        return 0, gap_1_2, gap_avg

def get_sorted_value_counts(value_counts):
    # Sort the dictionary items by value in descending order
    sorted_items = sorted(value_counts.items(), key=lambda item: item[1], reverse=True)

    # Separate the keys and values into two lists
    vals = [item[0] for item in sorted_items]
    counts = [item[1] for item in sorted_items]
    return vals, counts

def update_attack(als, res_key_prefix, filtered_df, precision_results, all_results):
    for mult in prediction_multipliers:
        res_key = f"{res_key_prefix}x{mult}"
        res_col = f"result_1_2_{mult}"
        df_predict = filtered_df[filtered_df[res_col] != 'abstain']
        if len(df_predict) == 0:
            precision_results[res_key] = {
                                    'prec_attack': None,
                                    'prec_base': None,
                                    'coverage': 0,
                                    'alc': 0,
                                    'num_rows': len(filtered_df),
                                    }
            all_results['summary'].append([res_key, 0])
            return
        df_true = filtered_df[filtered_df[res_col] == 'true']
        df_false = filtered_df[filtered_df[res_col] == 'false']
        coverage = (len(df_true) + len(df_false)) / len(filtered_df)
        prec_attack = len(df_true) / (len(df_true) + len(df_false))
        prec_base = 1 / (df_predict['num_vals'].sum() / len(df_predict))
        alc = als.alscore(p_base=prec_base, c_base=coverage, p_attack=prec_attack, c_attack=coverage)

        # Store the precision result
        precision_results[res_key] = {
                                    'prec_attack': prec_attack,
                                    'prec_base': prec_base,
                                    'coverage': coverage,
                                    'alc': alc,
                                    'num_rows': len(filtered_df),
                                    }
        all_results['summary'].append([res_key, alc])

class Attack1colResults:
    def __init__(self, path):
        self.parquet_path = os.path.join(path, f'1col_results.parquet')
        self.analyze_path = os.path.join(path, f'1col_results.json')
        if os.path.exists(self.parquet_path):
            self.df = pd.read_parquet(self.parquet_path)
            print(f"Loaded {len(self.df)} rows from {self.parquet_path}")
        else:
            self.df = pd.DataFrame(columns=[
                'num_vals', 'ex_factor', 'num_ex', 'dist', 'num_aid', 'predict_val', 'known_val', 'unknown_val', 'gap_1_2', 'gap_avg', 'value_counts'
            ])
        self.add_count = 0

    def add(self, num_vals, ex_factor, num_ex, dist, num_aid, predict_val, known_val, unknown_val, gap_1_2, gap_avg, value_counts):
        value_counts = json.dumps(value_counts)
        new_row = pd.DataFrame([{
            'num_vals': num_vals,
            'ex_factor': ex_factor,
            'num_ex': num_ex,
            'dist': dist,
            'num_aid': num_aid,
            'predict_val': predict_val,
            'known_val': known_val,
            'unknown_val': unknown_val,
            'gap_1_2': gap_1_2,
            'gap_avg': gap_avg,
            'value_counts': value_counts
        }])
        self.df = pd.concat([self.df, new_row], ignore_index=True)
        self.add_count += 1

        if self.add_count >= 100:
            self.write_to_parquet()
            self.add_count = 0

    def write_to_parquet(self):
        self.df.to_parquet(self.parquet_path, index=False)

    def make_predictions(self):
        def val_prediction(row, mult):
            # Example logic to assign a value
            value_counts = json.loads(row['value_counts'])
            # find the key with the highest corresponding value
            vals, counts = get_sorted_value_counts(value_counts)
            if counts[0] > counts[1] * mult:
                return vals[0]
            else:
                return 'abstain'
        def get_result(row, col_name):
            if row[col_name] == 'abstain':
                return 'abstain'
            elif row[col_name] == row['unknown_val']:
                return 'true'
            else:
                return 'false'
        for mult in prediction_multipliers:
            pred_col_name = f"pred_1_2_{mult}"
            self.df[pred_col_name] = self.df.apply(val_prediction, axis=1, args=(mult,))
            res_col_name = f"result_1_2_{mult}"
            self.df[res_col_name] = self.df.apply(get_result, axis=1, args=(pred_col_name,))

    def analyze(self):
        als = alscore.ALScore()
        # List of columns to analyze
        columns_to_analyze = ['num_vals', 'ex_factor', 'num_ex', 'num_aid', 'dist']

        all_results = {'precision_results': {}, 'summary':[]}
        
        # Dictionary to store the precision results
        precision_results = all_results['precision_results']
        
        for column in columns_to_analyze:
            # Get the distinct parameters in the column
            distinct_params = self.df[column].unique()
            for param in distinct_params:
                # Filter the DataFrame for the current column/param
                filtered_df = self.df[self.df[column] == param]
                res_key_prefix = f"{column}_{param}_"
                update_attack(als, res_key_prefix, filtered_df, precision_results, all_results)
        
        filtered_df = self.df[(self.df['ex_factor'] == 20) & (self.df['num_ex'] == 3) & (self.df['num_aid'] == 150)]
        res_key_prefix = f"ex_factor_20_num_ex_3_num_aid_150_"
        update_attack(als, res_key_prefix, filtered_df, precision_results, all_results)
        filtered_df = self.df[(self.df['ex_factor'] == 20) & (self.df['num_ex'] == 3) & (self.df['num_aid'] == 150) & (self.df['num_vals'] == 2)]
        res_key_prefix = f"ex_factor_20_num_ex_3_num_aid_150_num_vals_2_"
        update_attack(als, res_key_prefix, filtered_df, precision_results, all_results)
        # sort all_results['summary'] by the second element in each sublist
        all_results['summary'] = sorted(all_results['summary'], key=lambda x: x[1], reverse=True)
        # Write the precision results to a JSON file
        with open(self.analyze_path, 'w') as f:
            json.dump(all_results, f, indent=4)
        pp.pprint(all_results)

def run_1col_attack(job_num=None):
    a1c = Attack1colResults(results_path)
    for _ in range(num_1col_runs):
        for num_aid in [150, 500]:
            for dist in ['uniform', 'normal']:
                for num_vals in [2, 10]:
                    for ex_factor in [5, 10, 20]:
                        #for num_ex in [1, 2, 3]:
                        for num_ex in [3]:
                            run_one_1col_attack(a1c, num_vals, ex_factor, num_ex, dist, num_aid)
    a1c.write_to_parquet()


def gather_results():
    pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", help="'slurm' to make slurmscript, 'plot' to plot the results, 'attacks' to run all attacks, or an integer to run a specific attack")
    args = parser.parse_args()

    if args.command == 'slurm':
        make_slurm()
    elif args.command == 'plot':
        make_plot()
    elif args.command == 'attacks':
        run_1col_attack()
    elif args.command == 'gather':
        gather_results()
    elif args.command == 'membership':
        membership_attack()
    else:
        try:
            job_num = int(args.command)
            run_attack(job_num=job_num)
        except ValueError:
            print(f"Unknown command: {args.command}")

if __name__ == "__main__":
    main()