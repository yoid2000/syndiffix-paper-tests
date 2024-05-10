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

# Attacks

### Exact Count

This attack tries to determine the exact count of some aggregate (i.e. the number of columns with value V in column C). This is not an attack in and of itself, because an aggregate does not isolate a person per se. However, if it worked well, it could be a stepping stone to other attacks.

`exact_count.py` runs the exact_count measures. If run as `exact_count.py slurm`, it creates the file `exact_count_slurm.sh`. When `sbatch exact_count_slurm.sh` is run, 50 slurm jobs are started, each running `exact_count.py <slurm_job>`. This runs in a continuous loop, generating exact count attacks with randomly selected parameters, and adding the attack result to a file named `exact_count_results/results.XX.json`, where `XX` is the `<slurm_job>` number.

`exact_count_plot.py` reads in the `results.XX.json` files and generates a number of visual plots to help understand the results. These are placed in `results/exact_count/...`.

Besides the above, `exact_count_gather.py` and `exact_count_analyze.py` were used in the interim to help debug and understand the results. 

### Suppress Threshold Attack

This attack exploits the suppression mechanism of SynDiffix to try to determine if a given user has a certain value in some column. Under the conditions when the attacker knows that there are exactly three persons with a given value in column A, and knows that two of them have a certain other value in column B, the presence or absence of the column A value in the output allows the attacker, in certain situations, to determine whether the third person has the column B value.

This attack can be run with `suppress_threshold_theory.py`.

Requires two environment variables:

`SDX_TEST_DIR`: This is where the results are stored

`SDX_TEST_CODE`: This is the path to the directory where `suppress_threshold_theory.py` resides.

Requires that a virtual environment exists at `SDX_TEST_DIR/sdx_venv`

The syntax is:

`python suppress_threshold_theory.py <command>`

`command = slurm`: Generates a slurm file at `SDX_TEST_DIR/suppress_theory/theory.slurm`. Running `sbatch theory.slrum` will run the simulated attacks and place the results in a set of json files at `SDX_TEST_DIR/suppress_theory/tests`

`command = gather`: Reads in the json files at `SDX_TEST_DIR/suppress_theory/tests`, summarizes them, and and puts the summary in the file `SDX_TEST_DIR/suppress_theory/results/suppress_threshold_results.json`

`command = plot`: Creates a visual plot from `SDX_TEST_DIR/suppress_theory/results/suppress_threshold_results.json`, and puts it at `SDX_TEST_DIR/suppress_theory/results/suppress.png`.

`command = membership`: Runs a simple membership variant of the attack.  In this variant the attacker knows that there are either two or three persons in the dataset with a given column value, and knows that if there is a third person in the dataset then they must also have the column values, and wants to determine if the third person is indeed in the dataset.  Puts the result in `SDX_TEST_DIR/suppress_theory/results/suppress_threshold_results_no_infer.json`
