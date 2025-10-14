import json
import os
import argparse
import time
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaForSequenceClassification, get_linear_schedule_with_warmup # Changed Bert to Roberta
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, cohen_kappa_score
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default='data/paragraph_gt_test', help='Path to the data directory')
parser.add_argument('--model', type=str, default='FacebookAI/roberta-large', help='RoBERTa model name') # Changed default model
parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training and evaluation')
parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate for AdamW optimizer')
parser.add_argument('--max_len', type=int, default=512, help='Maximum sequence length for tokenizer')
parser.add_argument('--save_path', type=str, default='model/paragraph/roberta', help='Path to save the trained model') # Changed default save_path
args = parser.parse_args()

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

label_dict = {
'General': 0,
'Verify': 1,
'Explore': 2
}
num_labels = len(label_dict)

# Load Tokenizer
tokenizer = RobertaTokenizer.from_pretrained(args.model) # Changed BertTokenizer to RobertaTokenizer

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

# Load Data
all_texts = []
all_labels = []
print('Loading and preprocessing data...')
for file_name in sorted(os.listdir(args.path)):
    if file_name.endswith('.json'):
        print(f'Processing {file_name}...')
        with open(os.path.join(args.path, file_name), 'r') as f:
            data = json.load(f)
        for item in tqdm(data):
            all_texts.append(item['text'])
            all_labels.append(label_dict[item['gt-class-1']])

# Split data
train_texts, val_texts, train_labels, val_labels = train_test_split(
    all_texts, all_labels, test_size=0.3, random_state=42
)

train_dataset = TextClassificationDataset(train_texts, train_labels, tokenizer, args.max_len)
val_dataset = TextClassificationDataset(val_texts, val_labels, tokenizer, args.max_len)

train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
val_data_loader = DataLoader(val_dataset, batch_size=args.batch_size)

# Initialize Model
model = RobertaForSequenceClassification.from_pretrained( # Changed BertForSequenceClassification to RobertaForSequenceClassification
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
    num_warmup_steps=0, # Default to 0 warmup steps
    num_training_steps=total_steps
)

# Training Loop
print("Starting training...")
for epoch in range(args.epochs):
    print(f'Epoch {epoch + 1}/{args.epochs}')
    print('-' * 10)

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
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # Gradient clipping
        optimizer.step()
        scheduler.step()

    avg_train_loss = total_train_loss / len(train_data_loader)
    print(f'Training Loss: {avg_train_loss}')

    # Evaluation
    print("Evaluating...")
    model.eval()
    all_preds = []
    all_true_labels = []

    with torch.no_grad():
        for batch in tqdm(val_data_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(
                input_ids,
                attention_mask=attention_mask
            )

            logits = outputs.logits
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_true_labels.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(all_true_labels, all_preds)
    report = classification_report(
        all_true_labels, 
        all_preds, 
        target_names=label_dict.keys(),
        zero_division=0, # Added to handle cases with no predicted samples for a class
        output_dict=True
    )
    kappas = cohen_kappa_score(all_true_labels, all_preds)
    print(f'Validation Accuracy: {accuracy}')


# Save the model and tokenizer
print(f"Saving model to {args.save_path}")
if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)

model.save_pretrained(args.save_path)
tokenizer.save_pretrained(args.save_path)
print("Model and tokenizer saved.")


with open('method/paragraph/leaderboard.json', 'r') as f:
    leaderboard = json.load(f)

# Check if roberta entry already exists
roberta_exists = False
for entry in leaderboard:
    if entry['model'] == 'roberta':
        entry['accuracy'] = accuracy
        entry['kappas'] = kappas
        entry['report'] = report
        roberta_exists = True
        break

if not roberta_exists:
    leaderboard.append({
        'model': 'roberta',
        'accuracy': accuracy,
        'kappas': kappas,
        'report': report,
    })

with open('method/paragraph/leaderboard.json', 'w') as f:
    json.dump(leaderboard, f, indent=4)