# For the AI Voice Detector
# torchcodec uninstalled — HuggingFace falls back to soundfile automatically

import os
os.environ["FORCE_AUDIO_BACKEND"] = "soundfile"
os.environ["HF_DATASETS_AUDIO_BACKEND"] = "soundfile"

from pathlib import Path
import numpy as np
import soundfile as sf
from datasets import load_dataset, Audio


class VoiceDataDownloader:
    """Downloads real/fake audio using soundfile backend only."""

    def __init__(self):
        project_root = Path(__file__).resolve().parents[1]
        self.real_dir = project_root / "data" / "audio" / "real"
        self.fake_dir = project_root / "data" / "audio" / "fake"
        self.real_dir.mkdir(parents=True, exist_ok=True)
        self.fake_dir.mkdir(parents=True, exist_ok=True)

    def download(self) -> None:
        print("Downloading Voice dataset from HuggingFace...")

        dataset = load_dataset(
            "garystafford/deepfake-audio-detection",
            split="train",
            trust_remote_code=False,
        )

        # Force soundfile decoder — bypasses torchcodec entirely
        dataset = dataset.cast_column("audio", Audio(decode=False))

        print(f"Total samples: {len(dataset)}")
        real_count = 0
        fake_count = 0

        for i, sample in enumerate(dataset):
            try:
                # decode=False gives us raw file bytes + path
                audio_info = sample["audio"]
                audio_bytes = audio_info["bytes"]
                label = int(sample["label"])  # 0=real, 1=fake

                if label == 0:
                    out_path = self.real_dir / f"real_{real_count:04d}.wav"
                    real_count += 1
                else:
                    out_path = self.fake_dir / f"fake_{fake_count:04d}.wav"
                    fake_count += 1

                # Write raw bytes directly — no decoding needed
                with open(str(out_path), "wb") as f:
                    f.write(audio_bytes)

                if (i + 1) % 100 == 0:
                    print(f"  {i+1}/{len(dataset)} — real: {real_count}, fake: {fake_count}")

            except Exception as e:
                print(f"  Skipping sample {i}: {e}")
                continue

        print(f"\nDone! real: {real_count} files, fake: {fake_count} files")


if __name__ == "__main__":
    VoiceDataDownloader().download()