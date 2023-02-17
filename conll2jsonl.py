import csv
import json

from constants import non_tag, vac_tags


def convert_to_jsonlist(filename, tag_dict, delimiter='\t'):
    json_list = list()
    with open(filename, 'r') as csv_fh:
        csv_reader = csv.reader(csv_fh, delimiter='\t', quoting=csv.QUOTE_NONE)
        i = 0
        tokens = list()
        tags = list()
        prev_id = ''
        guid = 0
        for row in csv_reader:
            token, tag_str, _id = row[2], row[0], row[1]
            tag = tag_dict[tag_str]
            if prev_id == '':
                prev_id = _id
            if prev_id != _id:
                json_obj = {
                    "id": str(guid),
                    "tokens": tokens[:],
                    "ner_tags": tags[:],
                }
                json_list.append(json_obj)
                prev_id = _id
                tokens = [token]
                tags = [tag]
                guid += 1
            else:
                tokens.append(token)
                tags.append(tag)
        
        # the last doc
        json_obj = {
            "id": str(guid),
            "tokens": tokens[:],
            "ner_tags": tags[:],
        }
        json_list.append(json_obj)
        
    return json_list


def write_sample(json_list, output_file):
    with open(output_file, 'w') as json_fh:
        for json_obj in json_list:
            json.dump(json_obj, json_fh)
            json_fh.write('\n')


tags = build_tags(vac_tags, non_tag)
data_dir = 'data/vac-en/'

for split in ['train', 'devel', 'test']:
    json_list = convert_to_jsonlist(data_dir + '%s.tsv'%split, tags)
    write_sample(json_list, data_dir + '%s.jsonl'%split)