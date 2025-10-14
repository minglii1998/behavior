import os
import json

with open('Chenrui/sentence/7.json', 'r') as f:
    data_chenrui = json.load(f)

with open('Ming/sentence/7.json', 'r') as f:
    data_ming = json.load(f)

with open('Nan/sentence/7.json', 'r') as f:
    data_nan = json.load(f)

with open('gpt-4o/sentence/7.json', 'r') as f:
    data_gpt4o = json.load(f)

all_data = []
for i in range(len(data_chenrui)):
    new_data = {
        'id': i,
        'text': data_chenrui[i]['text'],
        'class_gpt4o': data_gpt4o[i]['class-2'],
        'class_chenrui': data_chenrui[i]['class-2'],
        'class_ming': data_ming[i]['class-2'],
        'class_nan': data_nan[i]['class-2'],
    }
    all_data.append(new_data)

os.makedirs('all_data/sentence', exist_ok=True)
with open('all_data/sentence/7.json', 'w') as f:
    json.dump(all_data, f, indent=4)