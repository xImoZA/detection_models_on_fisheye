import gdown
from ultralytics import RTDETR, SAM, YOLO

from src.app.utils.constants import MODEL_ID, MODEL_TYPE, MODEL_WEIGHTS


def download_from_google_drive(model: str) -> None:
    url = f"https://drive.google.com/uc?id={MODEL_ID[model]}"
    gdown.download(url, str(MODEL_WEIGHTS[model]), quiet=False)


def load_model(model_name: str) -> MODEL_TYPE:
    if model_name not in MODEL_WEIGHTS:
        raise ValueError(f"Unsupported model: {model_name}. Available: {list(MODEL_WEIGHTS.keys())}")

    weight_path = MODEL_WEIGHTS[model_name]
    weight_path.parent.mkdir(parents=True, exist_ok=True)

    if not weight_path.exists():
        download_from_google_drive(model_name)

    if model_name.startswith("YOLO"):
        return YOLO(str(weight_path))
    elif model_name == "RT-DETR":
        return RTDETR(str(weight_path))
    return SAM(str(weight_path))
