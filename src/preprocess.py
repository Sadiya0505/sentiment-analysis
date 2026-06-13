import pandas as pd
import re
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"<.*?>", "", text)        # remove HTML tags
    text = re.sub(r"[^a-z\s]", "", text)    # remove punctuation/numbers
    text = re.sub(r"\s+", " ", text).strip() # remove extra spaces
    return text

def preprocess(data_path: str = "data/raw/imdb.csv"):
    print("Loading data...")
    df = pd.read_csv(data_path)

    print("Cleaning text...")
    df["clean_text"] = df["text"].apply(clean_text)

    # Split into train/test
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    test_df  = df[df["split"] == "test"].reset_index(drop=True)

    print(f"Train size: {len(train_df):,} | Test size: {len(test_df):,}")

    # TF-IDF vectorization
    print("Fitting TF-IDF...")
    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
    X_train = vectorizer.fit_transform(train_df["clean_text"])
    X_test  = vectorizer.transform(test_df["clean_text"])

    y_train = train_df["label"].values
    y_test  = test_df["label"].values

    # Save everything
    os.makedirs("data/processed", exist_ok=True)
    joblib.dump(vectorizer, "data/processed/tfidf_vectorizer.pkl")
    joblib.dump((X_train, y_train), "data/processed/train.pkl")
    joblib.dump((X_test,  y_test),  "data/processed/test.pkl")

    print("Saved vectorizer and splits to data/processed/")
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    preprocess()