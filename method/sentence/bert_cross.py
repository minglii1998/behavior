import json
import os
import argparse
import time
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, cohen_kappa_score, confusion_matrix
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import entropy
from collections import Counter

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='google-bert/bert-base-uncased', help='BERT model name')
parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training and evaluation')
parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate for AdamW optimizer')
parser.add_argument('--max_len', type=int, default=512, help='Maximum sequence length for tokenizer')
parser.add_argument('--save_path', type=str, default='model/sentence/bert_cross', help='Path to save the trained model')
args = parser.parse_args()

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

label_dict = {
'Read': 0,
'Monitor': 1,
'Analyze': 2,
'Plan': 3,
'Implement': 4,
'Verify': 5,
'Explore': 6
}
num_labels = len(label_dict)
reverse_label_dict = {v: k for k, v in label_dict.items()}

# Load Tokenizer
tokenizer = BertTokenizer.from_pretrained(args.model)

# Define Dataset Class
class TextClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        label = self.labels[item]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def load_sat_data():
    """Load SAT dataset for training"""
    path = 'SAT/data/sentence_gt'
    all_texts = []
    all_labels = []
    print('Loading SAT training data...')
    for file_name in sorted(os.listdir(path)):
        if file_name.endswith('.json'):
            print(f'Processing {file_name}...')
            with open(os.path.join(path, file_name), 'r') as f:
                data = json.load(f)
            for item in tqdm(data):
                all_texts.append(item['text'])
                all_labels.append(label_dict[item['gt-class-2']])
    
    print(f"SAT dataset: {len(all_texts)} samples")
    return all_texts, all_labels

def load_omni_data():
    """Load Omni dataset for evaluation"""
    path_dirs = ['Omni/Omini_Nan_labels/deepseekR1/sentence', 
                 'Omni/Omini_Nan_labels/Phi4R/sentence', 
                 'Omni/Omini_Nan_labels/Qwen3_32B/sentence', 
                 'Omni/Omini_Nan_labels/QwQ32B/sentence']
    all_texts = []
    all_labels = []
    print('Loading Omni evaluation data...')
    for path_dir in path_dirs:
        for file_name in sorted(os.listdir(path_dir)):
            if file_name.endswith('.json'):
                print(f'Processing {path_dir}/{file_name}...')
                with open(os.path.join(path_dir, file_name), 'r') as f:
                    data = json.load(f)
                for item in tqdm(data):
                    try:
                        all_labels.append(label_dict[item['class-2']])
                        all_texts.append(item['text'])
                    except:
                        continue
    
    print(f"Omni dataset: {len(all_texts)} samples")
    return all_texts, all_labels

def analyze_label_distribution(labels, dataset_name):
    """Analyze and visualize label distribution"""
    label_counts = Counter(labels)
    label_names = [reverse_label_dict[label] for label in sorted(label_counts.keys())]
    counts = [label_counts[label] for label in sorted(label_counts.keys())]
    
    print(f"\n{dataset_name} Label Distribution:")
    for label_id, count in sorted(label_counts.items()):
        label_name = reverse_label_dict[label_id]
        percentage = (count / len(labels)) * 100
        print(f"  {label_name}: {count} ({percentage:.2f}%)")
    
    return label_counts

def calculate_distribution_divergence(train_dist, test_dist):
    """Calculate KL divergence and other distribution metrics"""
    # Normalize distributions
    train_probs = np.array([train_dist.get(i, 0) for i in range(num_labels)], dtype=float)
    test_probs = np.array([test_dist.get(i, 0) for i in range(num_labels)], dtype=float)
    
    train_probs = train_probs / train_probs.sum()
    test_probs = test_probs / test_probs.sum()
    
    # Add small epsilon to avoid log(0)
    eps = 1e-10
    train_probs = train_probs + eps
    test_probs = test_probs + eps
    
    # Normalize again after adding epsilon
    train_probs = train_probs / train_probs.sum()
    test_probs = test_probs / test_probs.sum()
    
    # Calculate KL divergence
    kl_div = entropy(test_probs, train_probs)
    
    # Calculate Jensen-Shannon divergence (symmetric)
    m = 0.5 * (train_probs + test_probs)
    js_div = 0.5 * (entropy(train_probs, m) + entropy(test_probs, m))
    
    # Calculate total variation distance
    tv_distance = 0.5 * np.sum(np.abs(train_probs - test_probs))
    
    print(f"\nDistribution Divergence Metrics:")
    print(f"  KL Divergence (Test||Train): {kl_div:.4f}")
    print(f"  Jensen-Shannon Divergence: {js_div:.4f}")
    print(f"  Total Variation Distance: {tv_distance:.4f}")
    
    return {
        'kl_divergence': kl_div,
        'js_divergence': js_div,
        'tv_distance': tv_distance,
        'train_distribution': train_probs.tolist(),
        'test_distribution': test_probs.tolist()
    }

def plot_distribution_comparison(train_dist, test_dist, save_path):
    """Plot and save distribution comparison"""
    labels = [reverse_label_dict[i] for i in range(num_labels)]
    train_counts = [train_dist.get(i, 0) for i in range(num_labels)]
    test_counts = [test_dist.get(i, 0) for i in range(num_labels)]
    
    # Normalize to percentages
    train_total = sum(train_counts)
    test_total = sum(test_counts)
    train_pcts = [count/train_total*100 for count in train_counts]
    test_pcts = [count/test_total*100 for count in test_counts]
    
    x = np.arange(len(labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width/2, train_pcts, width, label='SAT (Train)', alpha=0.8)
    bars2 = ax.bar(x + width/2, test_pcts, width, label='Omni (Test)', alpha=0.8)
    
    ax.set_xlabel('Labels')
    ax.set_ylabel('Percentage (%)')
    ax.set_title('Label Distribution Comparison: SAT (Train) vs Omni (Test)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45)
    ax.legend()
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)
    
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'distribution_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

# Load Data
print("="*50)
print("CROSS-DOMAIN EVALUATION: SAT → Omni")
print("="*50)

train_texts, train_labels = load_sat_data()
test_texts, test_labels = load_omni_data()

# Analyze label distributions
train_dist = analyze_label_distribution(train_labels, "SAT (Training)")
test_dist = analyze_label_distribution(test_labels, "Omni (Testing)")

# Split SAT data for training and validation
train_texts, val_texts, train_labels, val_labels = train_test_split(
    train_texts, train_labels, test_size=0.2, random_state=42
)

print(f"\nData split:")
print(f"  Training: {len(train_texts)} samples")
print(f"  Validation: {len(val_texts)} samples") 
print(f"  Test (Omni): {len(test_texts)} samples")

# Create datasets
train_dataset = TextClassificationDataset(train_texts, train_labels, tokenizer, args.max_len)
val_dataset = TextClassificationDataset(val_texts, val_labels, tokenizer, args.max_len)
test_dataset = TextClassificationDataset(test_texts, test_labels, tokenizer, args.max_len)

train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
val_data_loader = DataLoader(val_dataset, batch_size=args.batch_size)
test_data_loader = DataLoader(test_dataset, batch_size=args.batch_size)

# Initialize Model
model = BertForSequenceClassification.from_pretrained(
    args.model,
    num_labels=num_labels,
    output_attentions=False,
    output_hidden_states=False,
)
model.to(device)

# Optimizer and Scheduler
optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=1e-8)
total_steps = len(train_data_loader) * args.epochs
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
)

# Training Loop
print("\n" + "="*50)
print("TRAINING ON SAT DATASET")
print("="*50)

best_val_accuracy = 0
training_history = []

for epoch in range(args.epochs):
    print(f'\nEpoch {epoch + 1}/{args.epochs}')
    print('-' * 30)

    model.train()
    total_train_loss = 0

    for batch in tqdm(train_data_loader, desc="Training"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        model.zero_grad()

        outputs = model(
            input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss
        total_train_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

    avg_train_loss = total_train_loss / len(train_data_loader)
    
    # Validation on SAT data
    print("Evaluating on SAT validation set...")
    model.eval()
    val_preds = []
    val_true_labels = []

    with torch.no_grad():
        for batch in tqdm(val_data_loader, desc="Validating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(
                input_ids,
                attention_mask=attention_mask
            )

            logits = outputs.logits
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            val_preds.extend(preds)
            val_true_labels.extend(labels.cpu().numpy())
    
    val_accuracy = accuracy_score(val_true_labels, val_preds)
    
    # Save best model
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        print(f"New best validation accuracy: {val_accuracy:.4f}")
        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path)
        model.save_pretrained(args.save_path)
        tokenizer.save_pretrained(args.save_path)
    
    training_history.append({
        'epoch': epoch + 1,
        'train_loss': avg_train_loss,
        'val_accuracy': val_accuracy
    })
    
    print(f'Training Loss: {avg_train_loss:.4f}')
    print(f'Validation Accuracy: {val_accuracy:.4f}')

# Load best model for testing
print("\n" + "="*50)
print("LOADING BEST MODEL FOR CROSS-DOMAIN TESTING")
print("="*50)

model = BertForSequenceClassification.from_pretrained(args.save_path)
model.to(device)

# Test on Omni data
print("\n" + "="*50)
print("TESTING ON OMNI DATASET")
print("="*50)

model.eval()
test_preds = []
test_true_labels = []

with torch.no_grad():
    for batch in tqdm(test_data_loader, desc="Testing on Omni"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(
            input_ids,
            attention_mask=attention_mask
        )

        logits = outputs.logits
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        test_preds.extend(preds)
        test_true_labels.extend(labels.cpu().numpy())

# Calculate metrics
test_accuracy = accuracy_score(test_true_labels, test_preds)
test_kappa = cohen_kappa_score(test_true_labels, test_preds)
test_report = classification_report(
    test_true_labels, 
    test_preds, 
    target_names=label_dict.keys(),
    zero_division=0,
    output_dict=True
)

print(f'\n{"="*50}')
print("CROSS-DOMAIN EVALUATION RESULTS")
print("="*50)
print(f'Training Dataset: SAT')
print(f'Testing Dataset: Omni')
print(f'Test Accuracy: {test_accuracy:.4f}')
print(f'Test Cohen\'s Kappa: {test_kappa:.4f}')

# Calculate and save distribution divergence metrics
divergence_metrics = calculate_distribution_divergence(train_dist, test_dist)

# Create confusion matrix
cm = confusion_matrix(test_true_labels, test_preds)

# Save results and visualizations
results_dir = args.save_path
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# Plot distribution comparison
plot_distribution_comparison(train_dist, test_dist, results_dir)

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=label_dict.keys(), 
            yticklabels=label_dict.keys())
plt.title('Confusion Matrix: SAT-trained model on Omni data')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
plt.close()

# Save comprehensive results
results = {
    'experiment': 'cross_domain_sat_to_omni',
    'training_dataset': 'SAT',
    'testing_dataset': 'Omni',
    'model': 'bert',
    'training_samples': len(train_texts),
    'validation_samples': len(val_texts),
    'testing_samples': len(test_texts),
    'best_val_accuracy': best_val_accuracy,
    'test_accuracy': test_accuracy,
    'test_kappa': test_kappa,
    'test_report': test_report,
    'training_history': training_history,
    'divergence_metrics': divergence_metrics,
    'train_label_distribution': {reverse_label_dict[k]: v for k, v in train_dist.items()},
    'test_label_distribution': {reverse_label_dict[k]: v for k, v in test_dist.items()}
}

with open(os.path.join(results_dir, 'cross_domain_results.json'), 'w') as f:
    json.dump(results, f, indent=4)

print(f"\nResults saved to: {results_dir}")
print("Files created:")
print("  - cross_domain_results.json (comprehensive results)")
print("  - distribution_comparison.png (label distribution plot)")
print("  - confusion_matrix.png (confusion matrix heatmap)")

print(f'\n{"="*50}')
print("ANALYSIS SUMMARY")
print("="*50)
print(f"Distribution Divergence:")
print(f"  KL Divergence: {divergence_metrics['kl_divergence']:.4f}")
print(f"  JS Divergence: {divergence_metrics['js_divergence']:.4f}")
print(f"  TV Distance: {divergence_metrics['tv_distance']:.4f}")
print(f"\nPerformance Drop:")
print(f"  Best Validation Accuracy (SAT): {best_val_accuracy:.4f}")
print(f"  Test Accuracy (Omni): {test_accuracy:.4f}")
print(f"  Performance Drop: {(best_val_accuracy - test_accuracy):.4f}")

# Update leaderboard
leaderboard_path = 'method/sentence/leaderboard_cross.json'
if os.path.exists(leaderboard_path):
    with open(leaderboard_path, 'r') as f:
        leaderboard = json.load(f)
else:
    leaderboard = []

# Update or add entry
bert_exists = False
for entry in leaderboard:
    if entry['model'] == 'bert_sat_to_omni':
        entry.update({
            'test_accuracy': test_accuracy,
            'test_kappa': test_kappa,
            'test_report': test_report,
            'divergence_metrics': divergence_metrics
        })
        bert_exists = True
        break

if not bert_exists:
    leaderboard.append({
        'model': 'bert_sat_to_omni',
        'experiment': 'cross_domain_sat_to_omni',
        'test_accuracy': test_accuracy,
        'test_kappa': test_kappa,
        'test_report': test_report,
        'divergence_metrics': divergence_metrics
    })

with open(leaderboard_path, 'w') as f:
    json.dump(leaderboard, f, indent=4)

print(f"\nLeaderboard updated: {leaderboard_path}")
