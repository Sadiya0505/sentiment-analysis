import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from collections import Counter
import re
import os
import joblib

# ── 1. Tokenizer ──────────────────────────────────────────────
def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    return text.strip()

def build_vocab(texts, max_vocab=10000):
    counter = Counter()
    for t in texts:
        counter.update(t.split())
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for word, _ in counter.most_common(max_vocab - 2):
        vocab[word] = len(vocab)
    return vocab

def encode(text, vocab, max_len=200):
    tokens = text.split()[:max_len]
    ids = [vocab.get(t, 1) for t in tokens]
    ids += [0] * (max_len - len(ids))  # pad
    return ids

# ── 2. Dataset ─────────────────────────────────────────────────
class IMDBDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len=200):
        self.encodings = [encode(t, vocab, max_len) for t in texts]
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.encodings[idx], dtype=torch.long),
            torch.tensor(self.labels[idx],    dtype=torch.float),
        )

# ── 3. Model ───────────────────────────────────────────────────
class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=128, n_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm      = nn.LSTM(embed_dim, hidden_dim, n_layers,
                                 batch_first=True, dropout=0.3)
        self.dropout   = nn.Dropout(0.3)
        self.fc        = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        embedded = self.dropout(self.embedding(x))
        out, (hidden, _) = self.lstm(embedded)
        return self.fc(self.dropout(hidden[-1])).squeeze(1)

# ── 4. Train ───────────────────────────────────────────────────
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    df = pd.read_csv("data/raw/imdb.csv")
    df["clean"] = df["text"].apply(clean_text)

    train_df = df[df["split"] == "train"].reset_index(drop=True).sample(5000, random_state=42)
    test_df  = df[df["split"] == "test"].reset_index(drop=True).sample(1000, random_state=42)

    print("Building vocabulary...")
    vocab = build_vocab(train_df["clean"])
    joblib.dump(vocab, "models/lstm_vocab.pkl")

    train_ds = IMDBDataset(train_df["clean"], train_df["label"].values, vocab)
    test_ds  = IMDBDataset(test_df["clean"],  test_df["label"].values,  vocab)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=64)

    model     = SentimentLSTM(vocab_size=len(vocab)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    criterion = nn.BCEWithLogitsLoss()

    EPOCHS = 10
    for epoch in range(EPOCHS):
        model.train()
        total_loss, correct = 0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            loss  = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            correct    += ((preds > 0).float() == yb).sum().item()

        train_acc = correct / len(train_ds)
        # Validation
        model.eval()
        val_correct = 0
        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds  = model(xb)
                val_correct += ((preds > 0).float() == yb).sum().item()
        val_acc = val_correct / len(test_ds)

        print(f"Epoch {epoch+1}/{EPOCHS} | "
              f"Loss: {total_loss/len(train_loader):.4f} | "
              f"Train Acc: {train_acc:.4f} | "
              f"Val Acc: {val_acc:.4f}")

    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/lstm_model.pt")
    print("\nSaved LSTM model to models/lstm_model.pt")

if __name__ == "__main__":
    train()