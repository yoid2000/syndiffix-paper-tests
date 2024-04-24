import json

# read in the json file at results/exact_count_data.json
with open('./results/exact_count_data.json', 'r') as f:
    data = json.load(f)

for res in data:
    for stats in res['forest_stats'].values():
        pass
    data["num_val"]
    data["num_col"]
    data["dim"]
    data["num_row"]
    data["correct_averages"]