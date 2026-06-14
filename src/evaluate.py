import joblib
import torch
import pandas as pd
from sklearn.metrics import (accuracy_score, f1_score,
                             classification_report, roc_auc_score)
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import numpy as np

# ── 1. Evaluate Classical Models ───────────────────────────────
def evaluate_classical():
    print("=" * 50)
    print("CLASSICAL MODELS EVALUATION")
    print("=" * 50)

    X_test, y_test = joblib.load("data/processed/test.pkl")

    models = {
        "Logistic Regression": "models/logistic_regression.pkl",
        "Naive Bayes":         "models/naive_bayes.pkl",
        "Linear SVM":          "models/linear_svm.pkl",
    }

    results = []
    for name, path in models.items():
        model  = joblib.load(path)
        preds  = model.predict(X_test)
        acc    = accuracy_score(y_test, preds)
        f1     = f1_score(y_test, preds)
        print(f"\n{name}")
        print(classification_report(y_test, preds,
              target_names=["negative", "positive"]))
        results.append((name, acc, f1))

    return results

# ── 2. Evaluate BERT ───────────────────────────────────────────
def evaluate_bert():
    print("=" * 50)
    print("BERT EVALUATION")
    print("=" * 50)

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = DistilBertTokenizerFast.from_pretrained("models/bert")
    model     = DistilBertForSequenceClassification.from_pretrained("models/bert")
    model.to(device)
    model.eval()

    df       = pd.read_csv("data/raw/imdb.csv")
    test_df  = df[df["split"] == "test"].reset_index(drop=True)

    all_preds, all_labels = [], []
    batch_size = 16

    for i in range(0, len(test_df), batch_size):
        batch_texts  = test_df["text"].iloc[i:i+batch_size].tolist()
        batch_labels = test_df["label"].iloc[i:i+batch_size].tolist()

        inputs = tokenizer(batch_texts, truncation=True, padding=True,
                           max_length=256, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            preds   = outputs.logits.argmax(dim=1).cpu().numpy()

        all_preds.extend(preds)
        all_labels.extend(batch_labels)

        if (i // batch_size + 1) % 100 == 0:
            print(f"  Evaluated {i+batch_size}/{len(test_df)} samples...")

    acc = accuracy_score(all_labels, all_preds)
    f1  = f1_score(all_labels, all_preds)
    print(f"\nBERT Accuracy : {acc:.4f}")
    print(f"BERT F1 Score : {f1:.4f}")
    print(classification_report(all_labels, all_preds,
          target_names=["negative", "positive"]))
    return acc, f1

# ── 3. Final Comparison ────────────────────────────────────────
def print_comparison(classical_results, bert_acc, bert_f1):
    print("\n" + "=" * 50)
    print(f"{'Model':<25} {'Accuracy':>8} {'F1':>8}")
    print("=" * 50)
    for name, acc, f1 in classical_results:
        print(f"{name:<25} {acc:>8.4f} {f1:>8.4f}")
    print(f"{'DistilBERT':<25} {bert_acc:>8.4f} {bert_f1:>8.4f}")
    print("=" * 50)

if __name__ == "__main__":
    classical_results    = evaluate_classical()
    bert_acc, bert_f1    = evaluate_bert()
    print_comparison(classical_results, bert_acc, bert_f1)