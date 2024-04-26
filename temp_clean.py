import json
import os

json_files = [pos_json for pos_json in os.listdir("./exact_count_results") if pos_json.endswith('.json')]

data = []

for file in json_files:
    filepath = os.path.join("exact_count_results", file)
    with open(filepath, "r") as json_file:
        print(file)
        dat1 = json.load(json_file)
        for thing in dat1:
            if thing['num_col'] != 5 and thing['num_col'] != 6:
                data.append(thing)
    # write data to json file
    with open(filepath, "w") as json_file:
        json.dump(data, json_file)
