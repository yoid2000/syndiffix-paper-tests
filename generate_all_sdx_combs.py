import pandas as pd
import itertools
import json
import my_utilities as mu
import os

def makeSynFileName(baseName, cols):
    synFileName = baseName
    for col in cols:
        synFileName += f".{col}"
    return synFileName

maxComb = 3
baseDir = os.environ['SDX_TEST_DIR']
pqDir = os.path.join(baseDir, 'original_data_parquet')
allCombs = []
for fileName in [fileName for fileName in os.listdir(pqDir) if fileName.endswith('.parquet')]:
    baseName = fileName.replace('.parquet','')
    pqFilePath = os.path.join(pqDir, fileName)
    print(f"Read file {pqFilePath}")
    df = mu.load_pq(pqFilePath)
    columns = list(df.columns)
    for n_dims in range(1,maxComb+1):
        for comb in itertools.combinations(columns,n_dims):
            cols = sorted(list(comb))
            synFileName = makeSynFileName(baseName, cols)
            synFilePath = os.path.join(baseDir, baseName, synFileName + '.parquet')
            allCombs.append({'synDir':baseName,
                             'origFile': pqFilePath,
                             'synPath': synFilePath})
print(f"Made {len(allCombs)} combinations")
allCombsPath = os.path.join(baseDir, 'allSynCombs.json')
with open(allCombsPath, 'w') as f:
    print(f"Writing combinations to {allCombsPath}")
    json.dump(allCombs, f, indent=4)