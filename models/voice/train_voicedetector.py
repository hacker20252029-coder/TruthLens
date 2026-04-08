from pathlib import Path
from typing import List

import joblib
import librosa
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler


def extract_voice_features(audio_path: Path) -> np.ndarray:
    y, sr = librosa.load(str(audio_path), sr=None, mono=True)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfcc_means = np.mean(mfcc, axis=1)

    fmin = librosa.note_to_hz("C2")
    fmax = librosa.note_to_hz("C7")
    pitch = librosa.yin(y, fmin=fmin, fmax=fmax, sr=sr)
    pitch_mean = float(np.nanmean(pitch)) if pitch.size else 0.0
    pitch_var = float(np.nanvar(pitch)) if pitch.size else 0.0

    zcr = librosa.feature.zero_crossing_rate(y=y)
    zcr_mean = float(np.mean(zcr))

    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_centroid_mean = float(np.mean(spectral_centroid))

    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    spectral_rolloff_mean = float(np.mean(spectral_rolloff))

    rms = librosa.feature.rms(y=y)
    rms_var = float(np.var(rms))

    features = np.concatenate(
        [
            mfcc_means,
            np.array(
                [
                    pitch_mean,
                    pitch_var,
                    zcr_mean,
                    spectral_centroid_mean,
                    spectral_rolloff_mean,
                    rms_var,
                ],
                dtype=np.float32,
            ),
        ]
    )
    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def _collect_wav_files(directory: Path) -> List[Path]:
    return sorted([p for p in directory.glob("*.wav") if p.is_file()])


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    real_dir = project_root / "data" / "audio" / "real"
    fake_dir = project_root / "data" / "audio" / "fake"

    model_path = Path(__file__).resolve().parent / "voice_model.pkl"
    scaler_path = Path(__file__).resolve().parent / "voice_scaler.pkl"

    real_files = _collect_wav_files(real_dir)
    fake_files = _collect_wav_files(fake_dir)

    all_files = real_files + fake_files
    if not all_files:
        raise ValueError("No .wav files found in real/ or fake/ directories.")

    x_features = []
    y_labels = []

    for audio_file in real_files:
        x_features.append(extract_voice_features(audio_file))
        y_labels.append(0)  # real

    for audio_file in fake_files:
        x_features.append(extract_voice_features(audio_file))
        y_labels.append(1)  # fake/ai

    x = np.vstack(x_features)
    y = np.array(y_labels, dtype=np.int32)

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(x_scaled, y)

    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    print(f"Saved model to: {model_path}")
    print(f"Saved scaler to: {scaler_path}")


if __name__ == "__main__":
    main()
