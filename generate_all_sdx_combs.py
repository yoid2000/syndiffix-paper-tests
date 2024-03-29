import pandas as pd
import itertools
import json
import my_utilities as mu
import os
from pathlib import Path

DO_LOW_COMBS = False

maxComb = 3
slurmMem = '320G'
baseDir = os.environ['SDX_TEST_DIR']
pqDir = os.path.join(baseDir, 'original_data_parquet')

def updateAllCombs(synFilePath, allCombs, baseName, pqFilePath, cols):
    # check if the file at outPath already exists
    if os.path.exists(synFilePath):
        return
    index = len(allCombs)
    allCombs.append({'index':index,
                    'synDir':baseName,
                    'origFile': pqFilePath,
                    'synPath': synFilePath,
                    'cols': cols})

allCombs = []
for fileName in [fileName for fileName in os.listdir(pqDir) if fileName.endswith('.parquet')]:
    baseName = fileName.replace('.parquet','')
    pqFilePath = os.path.join(pqDir, fileName)
    print(f"Read file {pqFilePath}")
    df = mu.load_pq(pqFilePath)
    columns = list(df.columns)
    if DO_LOW_COMBS:
        for n_dims in range(1,maxComb+1):
            for comb in itertools.combinations(columns,n_dims):
                cols = sorted(list(comb))
                synFileName = mu.makeSynFileName(baseName, cols)
                synFilePath = os.path.join(baseDir, 'synDatasets', baseName,  synFileName + '.parquet')
                updateAllCombs(synFilePath, allCombs, baseName, pqFilePath, cols)
    synFileName = baseName + '.all'
    synFilePath = os.path.join(baseDir, 'synDatasets', baseName,  synFileName + '.parquet')
    updateAllCombs(synFilePath, allCombs, baseName, pqFilePath, columns)
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