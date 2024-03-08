import pandas as pd

'''
Read in the data in csv form, detect datetime columns, save as dataframes in SDX_TEST_DIR/original_data_df
'''

# Utility function for loading a CSV file.
def load_csv(path):
    from pandas.errors import ParserError

    df = pd.read_csv(path, keep_default_na=False, na_values=[""], low_memory=False)

    # Try to infer datetime columns.
    for col in df.columns[df.dtypes == "object"]:
        try:
            df[col] = pd.to_datetime(df[col], format="ISO8601")
        except (ParserError, ValueError):
            pass

    return df

def dump_pq(fileName, df):
    # Save to Parquet file
    df.to_parquet(fileName, engine='pyarrow')

def load_pq(fileName):
    # Load from Parquet file
    df = pd.read_parquet(fileName, engine='pyarrow')
    return df