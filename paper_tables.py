import pandas as pd
import itertools
import json
import my_utilities as mu
import os
import sys
from syndiffix_tools.tables_manager import TablesManager
from pathlib import Path

'''
Split the datasets into control and training parts
'''

baseDir = os.environ['SDX_TEST_DIR']
synDataPath = Path(baseDir, 'synDatasets')
tablesPath = Path(baseDir, 'paper_tables')
if not tablesPath.exists():
    tablesPath.mkdir()

for dir in os.listdir(synDataPath):
    thisDataPath = Path(synDataPath, dir)
    tm = TablesManager(dir_path=thisDataPath)
    pid_cols = tm.get_pid_cols()
    if tm.orig_meta_data['column_classes'][job['secret']] == 'continuous':
        pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", help="'config' to run make_config(), or an integer to run run_attacks()")
    args = parser.parse_args()

    if args.command == 'datasets':
        make_datasets()

if __name__ == "__main__":
    main()