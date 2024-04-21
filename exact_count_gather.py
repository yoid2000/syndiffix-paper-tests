import json
import os

# Step 1: Read in all of the .json files in directory ./exact_count_results
json_files = [pos_json for pos_json in os.listdir("./exact_count_results") if pos_json.endswith('.json')]

data_list = []

# Step 2: For each file, copy the specified items into a dict
for file in json_files:
    with open(f"./exact_count_results/{file}", "r") as json_file:
        data = json.load(json_file)
        data_dict = {key: data[key] for key in ("num_val", "num_col", "dim", "num_row", "correct_averages", "error_averages", "error_std_devs", "samples")}
        data_list.append(data_dict)

# Step 3: Put the dict into a list (already done in step 2)

# Step 4: Write the list of dicts into a json file in the local directory called exact_count_data.json
with open("exact_count_data.json", "w") as outfile:
    json.dump(data_list, outfile, indent=4)