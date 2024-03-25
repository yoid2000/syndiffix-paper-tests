import os
import sys
import json
from syndiffix.synthesizer import Synthesizer
import my_utilities as mu

baseDir = os.environ['SDX_TEST_DIR']
synDatasetsDir = os.path.join(baseDir, 'synDatasets')
os.makedirs(synDatasetsDir, exist_ok=True)
jobsFile = 'allSynCombs.json'
allCombsPath = os.path.join(baseDir, jobsFile)
with open(allCombsPath, 'r') as f:
    allCombs = json.load(f)
myJobNum = int(sys.argv[1])
if myJobNum > len(allCombs)-1:
    print(f"Bad job number {myJobNum}")
    quit()
job = allCombs[myJobNum]
inPath = job['origFile']
outPath = job['synPath']
columns = job['cols']
synDatasetDir = os.path.join(synDatasetsDir, job['synDir'])
os.makedirs(synDatasetDir, exist_ok=True)
print(job)
df = mu.load_pq(inPath)
df_syn = Synthesizer(df[columns]).sample()
mu.dump_pq(outPath, df_syn)