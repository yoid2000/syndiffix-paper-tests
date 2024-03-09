Code for testing syndiffix vulnerabilities, for the syndiffix paper zzzz

### Basic data prep

Requires python3.10 or greater.

Create a directory somewhere. Establish an environment variable `SDX_TEST_DIR` which is the path to that directory.

Under `SDX_TEST_DIR`, create a virtual environment called `sdx_venv`. (SLURM jobs assume this venv.)

Under `SDX_TEST_DIR`, create the directory `original_data_csv`, and populate it with whatever csv files you wish to test, as zipped files. For the paper we used the files in `original_data_csv`.

Copy the directory `original_data_csv` into SDX_TEST_DIR.

Run `csv_prep.py`. This creates `SDX_TEST_DIR/original_data_parquet` and populates it with dataframes from the csv files, with datetime columns typed as such.

### SynDiffix prep

Run `generate_all_sdx_combs.py`. This creates the file:
* `SDX_TEST_DIR/allSynCombs.json`, which contains the information needed to create every synthetic dataset, and
* `slurmGenSdx`, which is the slurm shell script needed to create the synthetic datasets

Run `sbatch slurmGenSdx`. This creates all of the SynDiffix synthetic datasets needed for our tests. This will consist of all 1dim, 2dim, and 3dim combinations of all of the test datasets.