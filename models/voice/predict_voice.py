from io import BytesIO
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Dict

import joblib
import librosa
import numpy as np


def extract_voice_features(audio_path: str) -> np.ndarray:
    y, sr = librosa.load(audio_path, sr=None, mono=True)

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


_MODEL_PATH = Path(__file__).resolve().parent / "voice_model.pkl"
_SCALER_PATH = Path(__file__).resolve().parent / "voice_scaler.pkl"

_MODEL = None
_SCALER = None

def _get_model():
    global _MODEL, _SCALER
    if _MODEL is None:
        _MODEL = joblib.load(_MODEL_PATH)
        _SCALER = joblib.load(_SCALER_PATH)
    return _MODEL, _SCALER


def _predict_from_features(features: np.ndarray) -> Dict[str, float | str]:
    x = features.reshape(1, -1)
    model, scaler = _get_model()
    x_scaled = scaler.transform(x)
    pred = int(model.predict(x_scaled)[0])
    proba = model.predict_proba(x_scaled)[0]

    confidence = float(proba[pred])
    label = "AI_VOICE" if pred == 1 else "REAL_VOICE"
    return {"label": label, "confidence": confidence}


def predict_voice(audio_path: str) -> Dict[str, float | str]:
    features = extract_voice_features(audio_path)
    return _predict_from_features(features)


def predict_voice_from_bytes(audio_bytes: bytes) -> Dict[str, float | str]:
    with NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
        temp_file.write(audio_bytes)
        tmp_path = temp_file.name
    try:
        features = extract_voice_features(tmp_path)
    finally:
        Path(tmp_path).unlink(missing_ok=True)
    return _predict_from_features(features)
