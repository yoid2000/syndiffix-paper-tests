import pandas as pd
import itertools
import json
import my_utilities as mu
import os
import sys
from syndiffix_tools.tables_manager import TablesManager
from pathlib import Path

'''
Go through all of the original datasets and define all possible combinations of columns with maxComb or fewer columns.
Also include synthetic dataset with all columns.
Put the results in allSynCombs.json and create the appropriate slurm scripts.
'''

DO_LOW_COMBS = True

maxComb = 3
baseDir = os.environ['SDX_TEST_DIR']
synDataPath = Path(baseDir, 'synDatasets')

if len(sys.argv) > 1:
    slurmMem = sys.argv[1]
else:
    slurmMem = '320G'

def updateAllCombs(allCombs, tm, cols):
    # check if the file at outPath already exists
    if tm.syn_file_exists(cols):
        return
    index = len(allCombs)
    allCombs.append({'index':index,
                    'synDir':tm.get_dir_path_str(),
                    'cols': cols})

allCombs = []
for dir in os.listdir(synDataPath):
    thisDataPath = Path(synDataPath, dir)
    print(f"Read file {thisDataPath}")
    tm = TablesManager(dir_path=thisDataPath)
    columns = list(tm.df_orig.columns)
    if DO_LOW_COMBS:
        for n_dims in range(1,maxComb+1):
            for comb in itertools.combinations(columns,n_dims):
                cols = sorted(list(comb))
                updateAllCombs(allCombs, tm, cols)
    updateAllCombs(allCombs, tm, columns)
print(f"Made {len(allCombs)} combinations")
allCombsPath = os.path.join(baseDir, 'allSynCombs.json')
with open(allCombsPath, 'w') as f:
    print(f"Writing combinations to {allCombsPath}")
    json.dump(allCombs, f, indent=4)

codeDir = Path().absolute()
sdxGenPath = os.path.join(codeDir, 'buildSdxDataset.py')
slurmPath = os.path.join(baseDir, 'slurmGenSdx')
testSlurmPath = os.path.join(baseDir, 'test_slurmGenSdx')
outputPath = os.path.join(baseDir, 'sdxOut')
os.makedirs(outputPath, exist_ok=True)
output = './sdxOut/out.%a.out'
slurmScript = f'''#!/bin/sh
#SBATCH --time=7-0
#SBATCH --array=0-10
#SBATCH --mem={slurmMem}
#SBATCH --output={output}
arrayNum="${{SLURM_ARRAY_TASK_ID}}"
source ./sdx_venv/bin/activate
python3 {sdxGenPath} $arrayNum
'''
with open(testSlurmPath, 'w') as f:
    f.write(slurmScript)

if len(allCombs) > 200:
    output = '/dev/null'
slurmScript = f'''#!/bin/sh
#SBATCH --time=7-0
#SBATCH --array=0-{len(allCombs)}
#SBATCH --mem={slurmMem}
#SBATCH --output={output}
arrayNum="${{SLURM_ARRAY_TASK_ID}}"
source ./sdx_venv/bin/activate
python3 {sdxGenPath} $arrayNum
'''
with open(slurmPath, 'w') as f:
    f.write(slurmScript)