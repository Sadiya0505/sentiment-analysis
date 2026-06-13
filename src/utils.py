from datasets import load_dataset
import pandas as pd
import os

def download_data(save_path: str = "data/raw/imdb.csv") -> pd.DataFrame:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    print("Downloading IMDB dataset...")
    dataset = load_dataset("stanfordnlp/imdb")

    train_df = dataset["train"].to_pandas()
    test_df  = dataset["test"].to_pandas()

    train_df["split"] = "train"
    test_df["split"]  = "test"

    df = pd.concat([train_df, test_df], ignore_index=True)
    df.to_csv(save_path, index=False)
    print(f"Saved {len(df):,} rows to {save_path}")
    return df

def explore_data(df: pd.DataFrame) -> None:
    print(f"\nShape: {df.shape}")
    print(f"\nClass distribution:\n{df['label'].value_counts()}")
    print(f"\nLabel map: 0=negative, 1=positive")
    print(f"\nAvg review length: {df['text'].str.split().str.len().mean():.0f} words")
    print(f"\nSample review:\n{df['text'].iloc[0][:300]}...")

if __name__ == "__main__":
    df = download_data()
    explore_data(df)