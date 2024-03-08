import my_utilities as mu
import os

''' Read in csv files, detect datetime fields and set appropriately, 
write as parquet dataframes
'''

base_dir = os.environ['SDX_TEST_DIR']
csv_path = os.path.join(base_dir, 'original_data_csv')
pq_path = os.path.join(base_dir, 'original_data_parquet')
os.makedirs(pq_path, exist_ok=True)

for filename in [filename for filename in os.listdir(csv_path) if filename.endswith('.csv')]:
    inpath = os.path.join(csv_path, filename)
    df = mu.load_csv(inpath)
    outpath = os.path.join(pq_path, filename)
    mu.dump_pq(outpath, df)