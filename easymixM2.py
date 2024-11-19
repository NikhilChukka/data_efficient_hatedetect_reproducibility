# CUDA Setup
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="1"

import torch
from torch import nn
from transformers import RobertaTokenizer, RobertaModel
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import classification_report, f1_score
from transformers import get_linear_schedule_with_warmup, AdamW
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel
#from torch.optim import Adam

# Set random seeds for reproducibility
np.random.seed(0)
torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)

# Basic setup
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print(f"Using device: {device}")


# Initialize tokenizer and constants
MAX_SEQ_LEN = 128
# tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
loss_fn = nn.CrossEntropyLoss()

class HateData(Dataset):
    def __len__(self):
        return len(self.data)

    def __init__(self, data_path, split='train', aug_prob=0.2, flip_prob=0.5):
        self.split = split
        self.data = pd.read_csv(data_path, sep='\t', lineterminator='\n')
        
        # Convert 3-class to 2-class format
        self.data['binary_label'] = self.data['label'].apply(lambda x: 1 if x == 1 else 0)
        
        print(f"\nLoading {split} data:")
        print("Original label distribution:")
        print(self.data['label'].value_counts())
        
        if self.split == 'train':
            self.label2data = {0:[], 1:[]}
            print("Creating binary label pools...")
            for i in tqdm(range(len(self.data))):
                row = self.data.iloc[i]
                self.label2data[row['binary_label']].append(row['post'])
            self.aug_prob = aug_prob
            self.flip_prob = flip_prob
            
            # Print label distribution
            print("\nOriginal label distribution:")
            print(self.data['label'].value_counts())
            print("\nBinary label distribution:")
            print(self.data['binary_label'].value_counts())

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        data = self.data.iloc[index]
        labels = data['binary_label']  # Use binary labels
        text = data['post']
        
        inputs = tokenizer(text, padding='max_length', truncation=True, max_length=MAX_SEQ_LEN)
        input_ids = inputs['input_ids']
        token_type_ids = np.zeros(MAX_SEQ_LEN)
        attn_mask = inputs['attention_mask']
        
        aug_text = text  
        labels_aug = labels
        
        if self.split == 'train' and labels == 1:
            if np.random.uniform() < self.aug_prob:
                aug_text = np.random.choice(self.label2data[0])
                
                if np.random.uniform() < self.flip_prob:
                    aug_text = aug_text + " [SEP] " + text
                else:
                    aug_text = text + " [SEP] " + aug_text
                labels_aug = 1
        
        inputs_aug = tokenizer(aug_text, padding='max_length', truncation=True, max_length=MAX_SEQ_LEN)
        input_ids_aug = inputs_aug['input_ids']
        token_type_ids_aug = np.zeros(MAX_SEQ_LEN)
        attn_mask_aug = inputs_aug['attention_mask']

        input_ids = torch.tensor(np.vstack([input_ids, input_ids_aug]), dtype=torch.long).view(2, MAX_SEQ_LEN)
        token_type_ids = torch.tensor(np.vstack([token_type_ids, token_type_ids_aug]), dtype=torch.long).view(2, MAX_SEQ_LEN)
        attn_mask = torch.tensor(np.vstack([attn_mask, attn_mask_aug]), dtype=torch.long).view(2, MAX_SEQ_LEN)
        labels = torch.tensor(np.vstack([labels, labels_aug]), dtype=torch.long).view(2)

        return input_ids, attn_mask, token_type_ids, labels

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        H1, H2 = 768, 128  # Hidden layer sizes
        
        # self.bert = RobertaModel.from_pretrained('roberta-base')
        self.bert = AutoModel.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
        
        
        self.clf = nn.Sequential(
            nn.Linear(H1, H2),
            nn.ReLU(),
            nn.Linear(H2, H2),
            nn.ReLU(),
            nn.Linear(H2, 2)  # Binary output
        )
        
    def forward(self, input_ids, attn_mask, token_type_ids):
        outputs = self.bert(input_ids, attn_mask)
        cls_emb = outputs.pooler_output
        logits = self.clf(cls_emb)
        return logits

def train(input_ids, attn_mask, token_type_ids, label, model, model_opt, scdl):
    model_opt.zero_grad()

    if use_cuda:
        input_ids = input_ids.to(device)
        attn_mask = attn_mask.to(device)
        token_type_ids = token_type_ids.to(device)
        label = label.to(device)

    # Debugging: Print shapes and types
    # print(f"Input IDs shape: {input_ids.shape}")
    # print(f"Attention Mask shape: {attn_mask.shape}")
    # print(f"Token Type IDs shape: {token_type_ids.shape}")
    # print(f"Labels shape: {label.shape}")
    # print(f"Input IDs dtype: {input_ids.dtype}")
    # print(f"Labels dtype: {label.dtype}")
    # print(f"Unique labels: {torch.unique(label)}")
    
    # Get model outputs
    logits = model(input_ids[:, 0, :], attn_mask[:, 0, :], token_type_ids[:, 0, :])
    logits_aug = model(input_ids[:, 1, :], attn_mask[:, 1, :], token_type_ids[:, 1, :])
    
    # Debugging: Print logits shape
    # print(f"Logits shape: {logits.shape}")
    
    # Convert labels to binary (0: non-hate, 1: hate)
    binary_labels = torch.where(label == 1, 1, 0)
    
    # Debugging: Print binary labels and their type
    # print(f"Binary Labels: {binary_labels}")
    # print(f"Binary Labels dtype: {binary_labels.dtype}")
    
    # Calculate loss with binary labels
    loss = loss_fn(logits, binary_labels[:, 0]) + loss_fn(logits_aug, binary_labels[:, 1])
    
    loss.backward()
    model_opt.step()
    scdl.step()
    
    return float(loss.item())

def evaluate(input_ids, attn_mask, token_type_ids, label, model, mode='train'):
    with torch.no_grad():
        if use_cuda:
            input_ids = input_ids.to(device)
            attn_mask = attn_mask.to(device)
            token_type_ids = token_type_ids.to(device)
            label = label.to(device)
        
        # Convert labels to binary
        binary_labels = torch.where(label == 1, 1, 0)
        
        logits = model(input_ids[:, 0, :], attn_mask[:, 0, :], token_type_ids[:, 0, :])
        loss = loss_fn(logits, binary_labels[:, 0])
        
        if mode == 'train':
            return float(loss.item())
        
        preds = torch.argmax(logits, dim=1).flatten()
        return float(loss.item()), preds.cpu().numpy()

def main():
    # Load your train and test data
    train_data = HateData(data_path="data/hatexplain/hx_train.tsv", split='train')
    val_data = HateData(data_path="data/hatexplain/hx_test.tsv", split='test')
    
    # Create dataloaders
    BS = 16
    train_loader = DataLoader(train_data, batch_size=BS, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BS, shuffle=False)
    
    # Load test labels
    df_test = pd.read_csv("data/hatexplain/hx_test.tsv", sep='\t', lineterminator='\n')
    global gt_labels
    gt_labels = np.array([1 if label == 1 else 0 for label in df_test['label']])
    
    # Initialize model
    model = Classifier().to(device)
    
    # Initialize optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=3e-5, eps=1e-8)
    num_training_steps = len(train_loader) * 5  # 5 epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * num_training_steps),
        num_training_steps=num_training_steps
    )
    
    print("Initialized optimizer and lr scheduler")
    
    best_f1 = 0
    tot = len(train_data) // train_loader.batch_size
    tot_val = len(val_data) // val_loader.batch_size
    
    for epoch in range(3):
        model.train()
        train_loss_total = 0.0
        train_step = 0
        
        progress_bar = tqdm(train_loader, total=tot, position=0, leave=True)
        for entry in progress_bar:
            loss = train(entry[0], entry[1], entry[2], entry[3], model, optimizer, scheduler)
            train_step += 1
            train_loss_total += loss
            
            train_loss = train_loss_total / train_step
            progress_bar.set_postfix({'loss': train_loss})
        
        # Validation
        model.eval()
        test_pred = []
        
        for entry in tqdm(val_loader, total=tot_val, position=0, leave=True):
            loss_v, pred_v = evaluate(entry[0], entry[1], entry[2], entry[3], model, mode='test')
            test_pred.extend([pd for pd in pred_v])
        
        # Calculate F1 score
        val_acc = f1_score(gt_labels, test_pred, average='macro')
        print(f"\nValidation F1: {val_acc:.4f}")
        
        # Print detailed classification report
        print("\nClassification Report:")
        print(classification_report(gt_labels, test_pred, digits=4))
        
        if val_acc > best_f1:
            torch.save(model.state_dict(), "best_model_roberta_hatexplain_easymix.pth")
            print("Model saved")
            best_f1 = val_acc
        
        print(f'Epoch: {epoch}')
        print(f'Total loss: {train_loss_total/tot:.4f}')

if __name__ == "__main__":
    main()