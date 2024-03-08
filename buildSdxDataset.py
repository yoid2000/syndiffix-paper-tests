import os
import sys
import json

baseDir = os.environ['SDX_TEST_DIR']
synDatasetsDir = os.path.join(baseDir, 'synDatasets')
os.makedirs(synDatasetsDir, exist_ok=True)
jobsFile = 'allSynCombs.json'
allCombsPath = os.path.join(baseDir, jobsFile)
with open(allCombsPath, 'r') as f:
    allCombs = json.load(f)
myJobNum = sys.argv[1]
if myJobNum > len(allCombs)-1:
    print(f"Bad job number {myJobNum}")
    quit()
job = allCombs[myJobNum]
inPath = job['origFile']
outPath = job['synPath']
synDatasetDir = os.path.join(synDatasetsDir, job['synDir'])
os.makedirs(synDatasetDir, exist_ok=True)
print(job)