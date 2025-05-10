from pathlib import Path
from typing import Union

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


GT_COLOR = (255, 105, 180)
