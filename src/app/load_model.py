from pathlib import Path
from typing import Union

import gdown
from ultralytics import RTDETR, SAM, YOLO

MODEL_TYPE = Union[YOLO, RTDETR, SAM]

MODEL_ID = {
    "YOLOv8": "1Is0UpjVOhN-iXQ2y8tlTU0b2ViP_7fQP",
    "YOLOv11": "1HGD0iDciTE_igPdJ4sFSUfRda33z5f31",
    "RT-DETR": "1WUMRtgcZC88lkaVJnt74FHs1E3wipw6K",
    "SAM": "1qNvOCZk9Cwa9qs-c6tOi3RPY7F26tseb",
}

MODEL_WEIGHTS = {
    "YOLOv8": Path("src/datasets/WoodScape/weights/yolov8m.pt"),
    "YOLOv11": Path("src/datasets/WoodScape/weights/yolov11m.pt"),
    "RT-DETR": Path("src/datasets/WoodScape/weights/rtdetr_l.pt"),
    "SAM": Path("src/datasets/WoodScape/weights/sam2.1_b.pt"),
}


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
