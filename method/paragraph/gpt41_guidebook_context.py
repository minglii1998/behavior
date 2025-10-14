import json
import os
from sklearn.metrics import accuracy_score, cohen_kappa_score, classification_report

path_dir = 'data/paragraph_llm'


all_data = []
for file in os.listdir(path_dir):
    if file.endswith('.json'):
        with open(os.path.join(path_dir, file), 'r') as f:
            data = json.load(f)
        all_data.extend(data)

predicted_labels = []
true_labels = []
for data in all_data:
    if data['gt-class-1'] == '':
        predicted_labels.append(data['class-1'])
        true_labels.append(data['class-1'])
    else:
        predicted_labels.append(data['class-1'])
        true_labels.append(data['gt-class-1'])

print('accuracy: ', accuracy_score(true_labels, predicted_labels))
kappas = cohen_kappa_score(true_labels, predicted_labels)
report = classification_report(true_labels, predicted_labels, output_dict=True)
with open('method/paragraph/leaderboard.json', 'r') as f:
    leaderboard = json.load(f)

gpt41_guidebook_context_exists = False
for entry in leaderboard:
    if entry['model'] == 'gpt41-guidebook-context':
        gpt41_guidebook_context_exists = True
        entry['kappas'] = kappas
        entry['accuracy'] = accuracy_score(true_labels, predicted_labels)
        entry['report'] = report
        break

if not gpt41_guidebook_context_exists:
    leaderboard.append({
        'model': 'gpt41-guidebook-context',
        'kappas': kappas,
        'accuracy': accuracy_score(true_labels, predicted_labels),
        'report': report
    })

with open('method/paragraph/leaderboard.json', 'w') as f:
    json.dump(leaderboard, f, indent=4)

# true_labels = [item['gt-class-1'] for item in all_data]
# predicted_labels = [item['class-1'] for item in all_data]

# all_labels = sorted(list(set(true_labels + predicted_labels)))

# from collections import defaultdict
# import matplotlib.pyplot as plt
# import seaborn as sns
# import numpy as np # For creating the matrix array
# confusion_matrix_dict = defaultdict(lambda: defaultdict(int))

# for true_label, predicted_label in zip(true_labels, predicted_labels):
#     confusion_matrix_dict[true_label][predicted_label] += 1

# # Convert defaultdict to a 2D list for heatmap
# matrix_size = len(all_labels)
# matrix_array = np.zeros((matrix_size, matrix_size), dtype=int)
# for i, true_label in enumerate(all_labels):
#     for j, pred_label in enumerate(all_labels):
#         matrix_array[i, j] = confusion_matrix_dict[true_label][pred_label]

# # Create an error matrix by zeroing out the diagonal (correct predictions)
# error_matrix_array = matrix_array.copy()
# for i in range(matrix_size):
#     error_matrix_array[i, i] = 0

# # --- Plotting Confusion Matrix as Heatmap ---
# plt.figure(figsize=(10, 8))
# sns.heatmap(error_matrix_array, annot=True, fmt="d", cmap="Blues",
#             xticklabels=all_labels, yticklabels=all_labels,
#             annot_kws={"size": 10}) # Adjust font size of annotations
# plt.title('Misclassification Heatmap', fontsize=15)
# plt.ylabel('True Label', fontsize=12)
# plt.xlabel('Predicted Label', fontsize=12)
# plt.xticks(rotation=45, ha='right')
# plt.yticks(rotation=0)
# plt.tight_layout() # Adjust layout to prevent labels from overlapping

# plt.savefig('data/paragraph_llm/confusion_matrix.png')

# target_dir = 'data/paragraph_llm_anno'
# for file in os.listdir(path_dir):
#     if file.endswith('.json'):
#         with open(os.path.join(path_dir, file), 'r') as f:
#             data = json.load(f)
#         for item in data:
#             if item['gt-class-1'] == '':
#                 item['gt-class-1'] = item['class-1']
#                 item['correct'] = True
#             else:
#                 item['correct'] = False
#         with open(os.path.join(target_dir, file), 'w') as f:
#             json.dump(data, f, indent=4)
