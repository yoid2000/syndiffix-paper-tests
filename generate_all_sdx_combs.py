import pandas as pd
import itertools
import json
import my_utilities as mu
import os
import sys
import random
from syndiffix_tools.tables_manager import TablesManager
from pathlib import Path

'''
Go through all of the original datasets and define all possible combinations of columns with maxComb or fewer columns.
Also include synthetic dataset with all columns.
Put the results in allSynCombs.json and create the appropriate slurm scripts.
'''

DO_LOW_COMBS = True
DO_FOUR_COMBS = True

maxComb = 3
numFourCombs = 30000
baseDir = os.environ['SDX_TEST_DIR']
synDataPath = Path(baseDir, 'synDatasets')

if len(sys.argv) > 1:
    slurmMem = sys.argv[1]
else:
    slurmMem = '10G'

def updateAllCombs(allCombs, tm, cols):
    # check if the file at outPath already exists
    if tm.syn_file_exists(cols):
        return 0
    index = len(allCombs)
    allCombs.append({'index':index,
                    'synDir':tm.get_dir_path_str(),
                    'cols': cols})
    return 1

allCombs = []
for dir in os.listdir(synDataPath):
    thisDataPath = Path(synDataPath, dir)
    tm = TablesManager(dir_path=thisDataPath)
    columns = list(tm.df_orig.columns)
    pid_cols = tm.get_pid_cols()
    # remove pid_cols from columns
    columns = [col for col in columns if col not in pid_cols]
    i = 0
    if DO_LOW_COMBS:
        for n_dims in range(1,maxComb+1):
            for comb in itertools.combinations(columns,n_dims):
                cols = sorted(list(comb))
                i += updateAllCombs(allCombs, tm, cols)
    i += updateAllCombs(allCombs, tm, columns)
    print(f"Created {i} combinations for {thisDataPath}")
if DO_FOUR_COMBS:
    j = 0
    alreadyHave = 0
    newCount = 0
    fourCombs = {}
    numFourCombDatasets = 0
    for dir in os.listdir(synDataPath):
        fourCombs[dir] = []
        thisDataPath = Path(synDataPath, dir)
        tm = TablesManager(dir_path=thisDataPath)
        columns = list(tm.df_orig.columns)
        pid_cols = tm.get_pid_cols()
        if len(pid_cols) > 0:
            continue
        numFourCombDatasets += 1
        for comb in itertools.combinations(columns,n_dims):
            cols = sorted(list(comb))
            if tm.syn_file_exists(cols):
                alreadyHave += 1
                continue
            newCount += 1
            fourCombs[dir].append([cols])
    print(f"Collected {newCount} fourCombs, {alreadyHave} already exist")
    needPerDataset = int((numFourCombs - newCount) / numFourCombDatasets)
    for dir in fourCombs.keys():
        thisDataPath = Path(synDataPath, dir)
        tm = TablesManager(dir_path=thisDataPath)
        colsList = random.sample(fourCombs[dir], min(needPerDataset, len(fourCombs[dir])))
        for cols in colsList:
            i += updateAllCombs(allCombs, tm, cols)
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