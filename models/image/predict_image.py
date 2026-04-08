from io import BytesIO
from pathlib import Path
from typing import Dict, Tuple

import torch
from PIL import Image, ExifTags
from torch import nn
from torchvision import models, transforms


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
)

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_MODEL_PATH = Path(__file__).resolve().parent / "image_model.pth"


def _build_model() -> nn.Module:
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 2)
    model.load_state_dict(torch.load(_MODEL_PATH, map_location=_DEVICE))
    model.to(_DEVICE)
    model.eval()
    return model


_MODEL = _build_model()


def _extract_exif(image: Image.Image) -> Dict[str, str]:
    exif_data = {}
    raw_exif = image.getexif()
    if not raw_exif:
        return exif_data
    for key, value in raw_exif.items():
        tag = ExifTags.TAGS.get(key, key)
        exif_data[str(tag)] = str(value)
    return exif_data


def _has_camera_data(exif_data: Dict[str, str]) -> bool:
    camera_fields = ["Make", "Model", "LensModel", "BodySerialNumber"]
    return any(field in exif_data and exif_data[field].strip() for field in camera_fields)


def _run_inference(image: Image.Image) -> Tuple[int, float]:
    image = image.convert("RGB")
    tensor = _TRANSFORM(image).unsqueeze(0).to(_DEVICE)

    with torch.no_grad():
        logits = _MODEL(tensor)
        probs = torch.softmax(logits, dim=1)[0]
        pred_idx = int(torch.argmax(probs).item())
        confidence = float(probs[pred_idx].item())
    return pred_idx, confidence


def _finalize_prediction(pred_idx: int, confidence: float, has_camera_data: bool) -> Dict[str, float | str]:
    # Class mapping follows ImageFolder alphabetical order: fake=0, real=1.
    ai_conf = confidence if pred_idx == 0 else 1.0 - confidence
    if not has_camera_data:
        ai_conf = min(1.0, ai_conf + 0.15)
    ai_conf = max(0.0, min(1.0, ai_conf))

    label = "AI_GENERATED" if ai_conf >= 0.5 else "REAL"
    final_conf = ai_conf if label == "AI_GENERATED" else 1.0 - ai_conf
    return {"label": label, "confidence": float(final_conf)}


def predict_image(image_path: str) -> Dict[str, float | str]:
    image = Image.open(image_path)
    exif_data = _extract_exif(image)
    has_camera_data = _has_camera_data(exif_data)
    pred_idx, confidence = _run_inference(image)
    return _finalize_prediction(pred_idx, confidence, has_camera_data)


def predict_image_from_bytes(image_bytes: bytes) -> Dict[str, float | str]:
    image = Image.open(BytesIO(image_bytes))
    exif_data = _extract_exif(image)
    has_camera_data = _has_camera_data(exif_data)
    pred_idx, confidence = _run_inference(image)
    return _finalize_prediction(pred_idx, confidence, has_camera_data)
