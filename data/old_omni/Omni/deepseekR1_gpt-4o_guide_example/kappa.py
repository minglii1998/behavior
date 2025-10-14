import json
import os
from sklearn.metrics import cohen_kappa_score

path_1 = 'Chenrui/paragraph/7.json'
path_2 = 'Nan/paragraph/7.json'

with open(path_1, 'r') as f:
    data_1 = json.load(f)

with open(path_2, 'r') as f:
    data_2 = json.load(f)

# Extract the 'class' annotations from both data_1 and data_2
labels_1 = [item['class'] for item in data_1]
labels_2 = [item['class'] for item in data_2]

# Ensure both lists are the same length
assert len(labels_1) == len(labels_2), "Annotation lists must be the same length."

# Calculate Cohen's kappa
kappa = cohen_kappa_score(labels_1, labels_2)
print(f"Cohen's kappa: {kappa} for paragraph")

path_1 = 'Chenrui/sentence/7.json'
path_2 = 'Nan/sentence/7.json'

with open(path_1, 'r') as f:
    data_1 = json.load(f)

with open(path_2, 'r') as f:
    data_2 = json.load(f)

# Extract the 'class' annotations from both data_1 and data_2
labels_1 = [item['class-2'] for item in data_1]
labels_2 = [item['class-2'] for item in data_2]

# Ensure both lists are the same length
assert len(labels_1) == len(labels_2), "Annotation lists must be the same length."

# Calculate Cohen's kappa
kappa = cohen_kappa_score(labels_1, labels_2)
print(f"Cohen's kappa: {kappa} for sentence")