import os
import sys
import json
from pathlib import Path
from syndiffix.synthesizer import Synthesizer
import my_utilities as mu

baseDir = os.environ['SDX_TEST_DIR']
# make sinDatasetsDir using Path
synDatasetsDir = Path(baseDir, 'synDatasets')
os.makedirs(synDatasetsDir, exist_ok=True)
jobsFile = 'allSynCombs.json'
allCombsPath = Path(baseDir, jobsFile)
with allCombsPath.open('r') as f:
    allCombs = json.load(f)
myJobNum = int(sys.argv[1])
if myJobNum > len(allCombs)-1:
    print(f"Bad job number {myJobNum}")
    quit()
job = allCombs[myJobNum]
inPath = job['origFile']
outPath = job['synPath']
# check if the file at outPath already exists
if os.path.exists(outPath):
    print(f"File {outPath} already exists. Skipping.")
    quit()
columns = job['cols']
synDatasetDir = os.path.join(synDatasetsDir, job['synDir'])
os.makedirs(synDatasetDir, exist_ok=True)
print(job)
df = mu.load_pq(inPath)
df_syn = Synthesizer(df[columns]).sample()
mu.dump_pq(outPath, df_syn)