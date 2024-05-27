import pandas as pd
import itertools
import json
import my_utilities as mu
import os
import sys
import random
import targets
from pathlib import Path

baseDir = os.environ['SDX_TEST_DIR']
synDataPath = Path(baseDir, 'synDatasets')

for dir in os.listdir(synDataPath):
    thisDataPath = Path(synDataPath, dir)
    baseFile = os.path.join(thisDataPath, f'{dir}.parquet')
    controlFile = os.path.join(thisDataPath, 'anonymeter', f'control.parquet')
    trainingFile = os.path.join(thisDataPath, 'anonymeter', f'training.parquet')

    if not os.path.exists(controlFile):
        print(f"baseFile {baseFile} does not exist")
        continue

    # read baseFile into dataframe
    df_base = pd.read_parquet(baseFile)
    df_control = pd.read_parquet(controlFile)
    df_training = pd.read_parquet(trainingFile)
    print(baseFile)
    print(f"baseFile: {df_base.shape}")
    print(controlFile)
    print(f"controlFile: {df_control.shape}")
    print(trainingFile)
    print(f"trainingFile: {df_training.shape}")