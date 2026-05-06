# For the Image Detector
# Dataset: Parveshiiii/AI-vs-Real
# Column: binary_label (0=real, 1=AI/fake) — confirmed from diagnostic run
# Total: 13,999 images

from pathlib import Path
from datasets import load_dataset


class ImageDataDownloader:
    """Downloads real vs AI-generated images into real/ and fake/ folders."""

    def __init__(self):
        project_root = Path(__file__).resolve().parents[1]
        self.real_dir = project_root / "data" / "images" / "real"
        self.fake_dir = project_root / "data" / "images" / "fake"
        self.real_dir.mkdir(parents=True, exist_ok=True)
        self.fake_dir.mkdir(parents=True, exist_ok=True)

    def download(self) -> None:
        print("Downloading Image dataset from HuggingFace...")
        print("Dataset: Parveshiiii/AI-vs-Real (13,999 images)")

        dataset = load_dataset(
            "Parveshiiii/AI-vs-Real",
            split="train",
            verification_mode="no_checks"
        )
        print(f"Total samples: {len(dataset)}")

        real_count = 0
        fake_count = 0

        for i, sample in enumerate(dataset):
            try:
                img          = sample["image"]
                binary_label = int(sample["binary_label"])  # 0=real, 1=fake

                if binary_label == 0:
                    out_path = self.real_dir / f"real_{real_count:05d}.jpg"
                    real_count += 1
                else:
                    out_path = self.fake_dir / f"fake_{fake_count:05d}.jpg"
                    fake_count += 1

                # Convert to RGB — handles RGBA/palette PNG images
                img.convert("RGB").save(str(out_path), "JPEG")

                if (i + 1) % 500 == 0:
                    print(f"  Processed {i+1}/{len(dataset)} "
                          f"— real: {real_count}, fake: {fake_count}")

            except Exception as e:
                print(f"  Skipping sample {i}: {e}")
                continue

        print(f"\nDone!")
        print(f"  Real images → {self.real_dir} ({real_count} files)")
        print(f"  Fake images → {self.fake_dir} ({fake_count} files)")


if __name__ == "__main__":
    ImageDataDownloader().download()