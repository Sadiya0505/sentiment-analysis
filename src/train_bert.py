import torch
import pandas as pd
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from torch.optim import AdamW
from transformers import get_scheduler
from sklearn.metrics import accuracy_score, f1_score

# ── 1. Dataset ─────────────────────────────────────────────────
class IMDBDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=256):
        self.encodings = tokenizer(
            list(texts),
            truncation=True,
            padding=True,
            max_length=max_len,
            return_tensors="pt"
        )
        self.labels = torch.tensor(list(labels), dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids":      self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels":         self.labels[idx]
        }

# ── 2. Train ───────────────────────────────────────────────────
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data — use small sample for CPU
    print("Loading data...")
    df = pd.read_csv("data/raw/imdb.csv")
    train_df = df[df["split"] == "train"].sample(3000, random_state=42)
    test_df  = df[df["split"] == "test"].sample(500,  random_state=42)

    print(f"Train: {len(train_df)} | Test: {len(test_df)}")

    # Load tokenizer and model
    print("Loading DistilBERT...")
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    model     = DistilBertForSequenceClassification.from_pretrained(
                    "distilbert-base-uncased", num_labels=2)
    model.to(device)

    # Tokenize
    print("Tokenizing...")
    train_ds = IMDBDataset(train_df["text"].values, train_df["label"].values, tokenizer)
    test_ds  = IMDBDataset(test_df["text"].values,  test_df["label"].values,  tokenizer)

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=8)

    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=2e-5)
    num_training_steps = 3 * len(train_loader)
    scheduler = get_scheduler("linear", optimizer=optimizer,
                               num_warmup_steps=0,
                               num_training_steps=num_training_steps)

    # Training loop
    EPOCHS = 3
    for epoch in range(EPOCHS):
        model.train()
        total_loss, correct = 0, 0

        for i, batch in enumerate(train_loader):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            outputs = model(input_ids=input_ids,
                           attention_mask=attention_mask,
                           labels=labels)
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            preds       = outputs.logits.argmax(dim=1)
            correct    += (preds == labels).sum().item()

            if (i + 1) % 50 == 0:
                print(f"  Step {i+1}/{len(train_loader)} | Loss: {loss.item():.4f}")

        train_acc = correct / len(train_ds)

        # Evaluation
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in test_loader:
                input_ids      = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels         = batch["labels"].to(device)
                outputs        = model(input_ids=input_ids,
                                      attention_mask=attention_mask)
                preds          = outputs.logits.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_acc = accuracy_score(all_labels, all_preds)
        val_f1  = f1_score(all_labels, all_preds)
        print(f"\nEpoch {epoch+1}/{EPOCHS} | "
              f"Loss: {total_loss/len(train_loader):.4f} | "
              f"Train Acc: {train_acc:.4f} | "
              f"Val Acc: {val_acc:.4f} | "
              f"Val F1: {val_f1:.4f}\n")

    # Save model
    os.makedirs("models/bert", exist_ok=True)
    model.save_pretrained("models/bert")
    tokenizer.save_pretrained("models/bert")
    print("Saved BERT model to models/bert/")

if __name__ == "__main__":
    train()