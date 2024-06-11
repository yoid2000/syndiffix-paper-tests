import json
import os
import sys
from syndiffix_tools.tables_manager import TablesManager
from pathlib import Path
import argparse
import pprint

pp = pprint.PrettyPrinter(indent=4)

baseDir = os.environ['SDX_TEST_DIR']
synDataPath = Path(baseDir, 'synDatasets')
tablesPath = Path(baseDir, 'paper_tables')
if not tablesPath.exists():
    tablesPath.mkdir()

def make_datasets():
    name_swaps = {
        'BankChurnersNoId': 'BankChurners',
    }
    table = '''
\\begin{table}
\\begin{center}
\\begin{small}
\\begin{tabular}{rllll}
\\toprule
Dataset & Rows & \\multirow{2}{*}{Columns} & Time Series  \\\\
  &  & Cat & Con &  \\\\
\\midrule
'''
    for dir in os.listdir(synDataPath):
        thisDataPath = Path(synDataPath, dir)
        tm = TablesManager(dir_path=thisDataPath)
        pid_cols = tm.get_pid_cols()
        print("--------------------------------------------------------")
        omd = tm.orig_meta_data
        table_name = omd['orig_file_name'][:-8]
        if table_name in name_swaps:
            table_name = name_swaps[table_name]
        print(f"Table: {table_name}")
        if len(pid_cols) > 0:
            time_series = 'yes'
        else:
            time_series = 'no'
        num_rows = omd['num_rows']
        print(f"Number of rows: {num_rows}")
        num_cols = omd['num_cols']
        print(f"Number of columns: {num_cols}")
        num_cat = 0
        num_con = 0
        for val in omd['column_classes'].values():
            if val == 'categorical':
                num_cat += 1
            else:
                num_con += 1
        print(f"Number of categorical columns: {num_cat}")
        print(f"Number of continuous columns: {num_con}")
        print(f"Time series: {time_series}")
        table += f"{table_name} & {num_rows} & {num_cat} & {num_con} & {time_series} \\\\ \n"
    table += '''
\\bottomrule
\\end{tabular}
\\end{small}
\\end{center}
\\caption{Synthetic datasets used in the experiments.}
\\label{tab:syn_datasets}
\\end{table}
'''
    print(f"Writing table to {Path(tablesPath, 'syn_datasets.tex')}")
    with open(Path(tablesPath, 'syn_datasets.tex'), 'w') as f:
        f.write(table)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", help="'config' to run make_config(), or an integer to run run_attacks()")
    args = parser.parse_args()

    if args.command == 'datasets':
        make_datasets()

if __name__ == "__main__":
    main()