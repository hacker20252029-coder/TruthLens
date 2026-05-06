# For the Fake News Detector
# Downloads fake + real news → data/text/news.csv (columns: text, label)
# label: 0=real, 1=fake

from pathlib import Path
from datasets import load_dataset
import pandas as pd


class FakeNewsDataDownloader:
    """Downloads fake news dataset and saves as news.csv."""

    def __init__(self):
        project_root = Path(__file__).resolve().parents[1]
        self.out_path = project_root / "data" / "text" / "news.csv"
        self.out_path.parent.mkdir(parents=True, exist_ok=True)

    def download(self) -> None:
        print("Downloading Fake News dataset from HuggingFace...")
        # GonzaloA/fake_news has columns: text, label (0=real, 1=fake)
        dataset = load_dataset("GonzaloA/fake_news", split="train")
        print(f"Total samples: {len(dataset)}")

        # Convert to pandas dataframe
        df = dataset.to_pandas()

        # Keep only text and label columns
        df = df[["text", "label"]].dropna()
        df["label"] = df["label"].astype(int)

        df.to_csv(str(self.out_path), index=False)
        print(f"\nDone! Saved {len(df)} rows to {self.out_path}")
        print(f"  Real (0): {len(df[df['label'] == 0])}")
        print(f"  Fake (1): {len(df[df['label'] == 1])}")


if __name__ == "__main__":
    FakeNewsDataDownloader().download()