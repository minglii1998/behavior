import json
import os
import argparse
import time
from method.utils import get_embedding
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, cohen_kappa_score
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default='data/sentence_gt')
parser.add_argument('--model', type=str, default='gemini')
parser.add_argument('--hidden_size', type=int, default=128, help='Hidden size for MLP')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for MLP')
parser.add_argument('--epochs', type=int, default=50, help='Number of epochs for MLP training')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size for MLP training')
args = parser.parse_args()

path = args.path
model_name = args.model # Renamed to avoid conflict with PyTorch model
hidden_size = args.hidden_size
lr = args.lr
epochs = args.epochs
batch_size = args.batch_size

label_dict = {
    'Read': 0,
    'Monitor': 1,
    'Analyze': 2,
    'Plan': 3,
    'Implement': 4,
    'Verify': 5,
    'Explore': 6
}
num_classes = len(label_dict)

all_data = []
embedding_path = f'embedding/{path.split("/")[-1]}_{model_name}_embedding.json'
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
            # Assuming get_embedding returns a list/np.array of numbers
            embedding = get_embedding(text, model_name)
            time.sleep(1) # Respect API limits if any
            new_item = {
                'embedding': embedding,
                'label': label_dict[item['gt-class-1']]
            }
            all_data.append(new_item)
    with open(embedding_path, 'w') as f:
        json.dump(all_data, f)
    print(f'Embeddings saved to {embedding_path}')

if not all_data:
    print("No data loaded. Exiting.")
    exit()

input_size = len(all_data[0]['embedding']) # Determine input size from the first embedding

train_data, test_data = train_test_split(all_data, test_size=0.2, random_state=42)

X_train_list = [item['embedding'] for item in train_data]
y_train_list = [item['label'] for item in train_data]
X_test_list = [item['embedding'] for item in test_data]
y_test_list = [item['label'] for item in test_data]

# Convert to PyTorch tensors
X_train = torch.tensor(X_train_list, dtype=torch.float32)
y_train = torch.tensor(y_train_list, dtype=torch.long)
X_test = torch.tensor(X_test_list, dtype=torch.float32)
y_test = torch.tensor(y_test_list, dtype=torch.long)

train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Define the MLP model
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

mlp_model = MLP(input_size, hidden_size, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(mlp_model.parameters(), lr=lr)

# Training loop
print(f"Starting MLP training for {epochs} epochs...")
for epoch in range(epochs):
    mlp_model.train()
    running_loss = 0.0
    for i, (embeddings, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = mlp_model(embeddings)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}")

# Evaluation
mlp_model.eval()
y_pred_list = []
y_true_list = []
with torch.no_grad():
    for embeddings, labels in test_loader:
        outputs = mlp_model(embeddings)
        _, predicted = torch.max(outputs.data, 1)
        y_pred_list.extend(predicted.cpu().numpy())
        y_true_list.extend(labels.cpu().numpy())

accuracy = accuracy_score(y_true_list, y_pred_list)
report = classification_report(y_true_list, y_pred_list, target_names=label_dict.keys(), output_dict=True, zero_division=0)
kappas = cohen_kappa_score(y_true_list, y_pred_list)

print(f"MLP Hyperparameters: hidden_size={hidden_size}, lr={lr}, epochs={epochs}, batch_size={batch_size}")
print(f"Accuracy: {accuracy:.4f}")


leaderboard_path = 'method/sentence/leaderboard.json'
if os.path.exists(leaderboard_path):
    with open(leaderboard_path, 'r') as f:
        leaderboard = json.load(f)
else:
    leaderboard = []


model_identifier = f'mlp-{model_name}'
mlp_exists = False
for entry in leaderboard:
    if entry['model'] == model_identifier:
        entry['accuracy'] = accuracy
        entry['report'] = report
        entry['kappas'] = kappas
        mlp_exists = True
        break

if not mlp_exists:
    leaderboard.append({
        'model': model_identifier,
        'accuracy': accuracy,
        'kappas': kappas,
        'report': report
    })

with open(leaderboard_path, 'w') as f:
    json.dump(leaderboard, f, indent=4)

print(f"Results saved to {leaderboard_path}") 