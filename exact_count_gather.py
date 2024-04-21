import json
import os

json_files = [pos_json for pos_json in os.listdir("./exact_count_results") if pos_json.endswith('.json')]

data_list = []

for file in json_files:
    with open(f"./exact_count_results/{file}", "r") as json_file:
        data = json.load(json_file)
        data_dict = {key: data[key] for key in ("num_val", "num_col", "dim", "num_row", "correct_averages", "error_averages", "error_std_devs", "samples")}
        data_list.append(data_dict)
# create directory 'results' if it doesn't exist
if not os.path.exists("results"):
    os.makedirs("results")
out_path = os.path.join("results", "exact_count_data.json")

with open(out_path, "w") as outfile:
    json.dump(data_list, outfile, indent=4)