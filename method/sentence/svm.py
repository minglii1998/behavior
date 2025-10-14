import json
import os
import argparse
import time
from method.utils import get_embedding
from tqdm import tqdm
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, cohen_kappa_score

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default='data/sentence_gt')
parser.add_argument('--model', type=str, default='gemini')
parser.add_argument('--kernel', type=str, default='rbf', help='Kernel type for SVC (e.g., linear, rbf, poly)')
args = parser.parse_args()

path = args.path
model = args.model
kernel = args.kernel
label_dict = {
    'Read': 0,
    'Monitor': 1,
    'Analyze': 2,
    'Plan': 3,
    'Implement': 4,
    'Verify': 5,
    'Explore': 6
}

all_data = []
embedding_path = f'embedding/{path.split("/")[-1]}_{model}_embedding.json'
if not os.path.exists(embedding_path):
    os.makedirs(os.path.dirname(embedding_path), exist_ok=True)
    not_exist = True
else:
    print('Embedding exist, start to load...')
    with open(embedding_path, 'r') as f:
        all_data = json.load(f)
    not_exist = False

if not_exist:
    print('Embedding not exist, start to embed...')
    for file in sorted(os.listdir(path)):
        print(f'Processing {file}...')
        with open(os.path.join(path, file), 'r') as f:
            data = json.load(f)
        for item in tqdm(data):
            text = item['text']
            embedding = get_embedding(text, model)
            time.sleep(1)
            new_item = {
                'embedding': embedding,
                'label': label_dict[item['gt-class-1']]
            }
            all_data.append(new_item)
    with open(embedding_path, 'w') as f:
        json.dump(all_data, f)
    print(f'Embeddings saved to {embedding_path}')

train_data, test_data = train_test_split(all_data, test_size=0.2, random_state=42)

X_train = [item['embedding'] for item in train_data]
y_train = [item['label'] for item in train_data]
X_test = [item['embedding'] for item in test_data]
y_test = [item['label'] for item in test_data]

# Initialize and train the Kernel SVM classifier
svm_classifier = SVC(kernel=kernel)
svm_classifier.fit(X_train, y_train)

y_pred = svm_classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=label_dict.keys(), output_dict=True)
kappas = cohen_kappa_score(y_test, y_pred)

print(f"Kernel: {kernel}")
print(f"Accuracy: {accuracy}")

with open('method/sentence/leaderboard.json', 'r') as f:
    leaderboard = json.load(f)

svm_exists = False
for entry in leaderboard:
    if entry['model'] == f'svm-{args.model}-{kernel}':
        entry['accuracy'] = accuracy
        entry['report'] = report
        entry['kappas'] = kappas
        svm_exists = True
        break

if not svm_exists:
    leaderboard.append({
        'model': f'svm-{args.model}-{kernel}',
        'accuracy': accuracy,
        'kappas': kappas,
        'report': report
    })

with open('method/sentence/leaderboard.json', 'w') as f:
    json.dump(leaderboard, f, indent=4)