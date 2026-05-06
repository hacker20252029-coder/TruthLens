# For the AI Text Detector — fixed version
# Uses different HuggingFace dataset that works without legacy scripts

from pathlib import Path
from datasets import load_dataset
import pandas as pd


class AITextDataDownloader:
    """Downloads AI vs Human text dataset and saves as ai_vs_human.csv."""

    def __init__(self):
        project_root = Path(__file__).resolve().parents[1]
        self.out_path = project_root / "data" / "text" / "ai_vs_human.csv"
        self.out_path.parent.mkdir(parents=True, exist_ok=True)

    def download(self) -> None:
        print("Downloading AI Text dataset from HuggingFace...")

        # Using a dataset that works without legacy scripts
        dataset = load_dataset(
            "artem9k/ai-text-detection-pile",
            split="train"
        )
        print(f"Total samples: {len(dataset)}")

        df = dataset.to_pandas()

        # Rename columns to match what train_aidetector.py expects
        df = df.rename(columns={"source": "label"})

        # Convert labels: 0=human, 1=AI
        df["label"] = df["label"].apply(lambda x: 1 if str(x).lower() == "ai" else 0)
        df = df[["text", "label"]].dropna()

        df.to_csv(str(self.out_path), index=False)
        print(f"\nDone! Saved {len(df)} rows to {self.out_path}")
        print(f"  Human (0): {len(df[df['label'] == 0])}")
        print(f"  AI    (1): {len(df[df['label'] == 1])}")


if __name__ == "__main__":
    AITextDataDownloader().download()