import os
import json
import plotly

from utils import tag_list, vac_tags

data_folder = 'data/vac-nl'
tag_list = [k for k in tag_list.keys()]

for json_file in ['train-all.jsonl']:
    json_path = os.path.join(data_folder, json_file)
    with open(json_path, 'r') as json_fr:
        json_list = json_fr.readlines()
        counter = {tag:0 for tag in vac_tags}
        for json_str in json_list:
            json_obj = json.loads(json_str)
            for tag in json_obj['ner_tags']:
                if tag != 0:
                    counter[tag_list[tag].replace('open', '')] += 1
        print(counter)