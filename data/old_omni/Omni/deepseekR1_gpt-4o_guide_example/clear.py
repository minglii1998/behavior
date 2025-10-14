import json
import os

path = 'Chenrui/paragraph'

for file in os.listdir(path):
    with open(os.path.join(path, file), 'r') as f:
        data = json.load(f)
        for item in data:
            item['class'] = ''
            # del item['class-2-reason']
            del item['reason']

    with open(os.path.join(path, file), 'w') as f:
        json.dump(data, f, indent=4)
