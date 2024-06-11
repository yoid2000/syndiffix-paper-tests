import json
import os
import sys
from syndiffix_tools.tables_manager import TablesManager
from pathlib import Path
import pprint

pp = pprint.PrettyPrinter(indent=4)

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
    pp.pprint(tm.orig_meta_data)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", help="'config' to run make_config(), or an integer to run run_attacks()")
    args = parser.parse_args()

    if args.command == 'datasets':
        make_datasets()

if __name__ == "__main__":
    main()