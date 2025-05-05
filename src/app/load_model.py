from pathlib import Path
from typing import Union

from ultralytics import RTDETR, SAM, YOLO

MODEL_TYPE = Union[YOLO, RTDETR, SAM]

MODEL_WEIGHTS = {
    "YOLOv8": "src/models/weights/yolov8m.pt",
    "YOLOv11": "src/models/weights/yolov11m.pt",
    "RT-DETR": "src/models/weights/rtdetr_l.pt",
    "SAM": "src/models/weights/sam2.1_b.pt",
}


def load_model(model_name: str):
    if model_name not in MODEL_WEIGHTS:
        raise ValueError(
            f"Unsupported model: {model_name}. Available: {list(MODEL_WEIGHTS.keys())}"
        )

    weight_path = Path(MODEL_WEIGHTS[model_name])
    if not weight_path.exists():
        raise FileNotFoundError(f"Model weights not found at: {weight_path}")

    if model_name.startswith("YOLO"):
        return YOLO(str(weight_path))
    elif model_name == "RT-DETR":
        return RTDETR(str(weight_path))
    return SAM(str(weight_path))
