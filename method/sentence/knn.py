import json
import os
import argparse
import time
from method.utils import get_embedding
from tqdm import tqdm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import cohen_kappa_score

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default='data/sentence_gt')
parser.add_argument('--model', type=str, default='gemini')
parser.add_argument('--n_neighbors', type=int, default=3)
args = parser.parse_args()

path = args.path
model = args.model
n_neighbors = args.n_neighbors
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
    for idx, file in enumerate(sorted(os.listdir(path))):        
        print(f'Processing {file}...')
        with open(os.path.join(path, file), 'r') as f:
            data = json.load(f)
        for item in tqdm(data):
            text = item['text']
            embedding = get_embedding(text, model)
            time.sleep(1)
            new_item = {
                'embedding': embedding,
                'label': label_dict[item['gt-class-2']]
            }
            all_data.append(new_item)
        if idx % 5 == 0:
            with open(embedding_path, 'w') as f:
                json.dump(all_data, f)
            print(f'Embeddings saved to {embedding_path}')
    # Save the newly computed embeddings
    with open(embedding_path, 'w') as f:
        json.dump(all_data, f)
    print(f'Embeddings saved to {embedding_path}')

print(len(all_data))
from sklearn.model_selection import train_test_split
train_data, test_data = train_test_split(all_data, test_size=0.2, random_state=42)

# Prepare data for scikit-learn
X_train = [item['embedding'] for item in train_data]
y_train = [item['label'] for item in train_data]
X_test = [item['embedding'] for item in test_data]
y_test = [item['label'] for item in test_data]

# Initialize and train the k-NN classifier
knn = KNeighborsClassifier(n_neighbors=n_neighbors) # You can tune n_neighbors
knn.fit(X_train, y_train)

# Make predictions on the test set
y_pred = knn.predict(X_test)

# Evaluate the model
from sklearn.metrics import accuracy_score, classification_report

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=label_dict.keys(), output_dict=True)
kappas = cohen_kappa_score(y_test, y_pred)

print(f"Accuracy: {accuracy}")

with open('method/paragraph/leaderboard.json', 'r') as f:
    leaderboard = json.load(f)

knn_exists = False
for entry in leaderboard:
    if entry['model'] == f'knn-{args.model}-{n_neighbors}':
        entry['accuracy'] = accuracy
        entry['report'] = report
        entry['kappas'] = kappas
        knn_exists = True
        break

if not knn_exists:
    leaderboard.append({
        'model': f'knn-{args.model}-{n_neighbors}',
        'accuracy': accuracy,
        'kappas': kappas,
        'report': report
    })

with open('method/sentence/leaderboard.json', 'w') as f:
    json.dump(leaderboard, f, indent=4)