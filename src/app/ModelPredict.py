from typing import Union

import cv2
from ultralytics import RTDETR, SAM, YOLO
from ultralytics.engine.results import Results

Model = Union[YOLO, RTDETR, SAM]


class ModelPredict:
    def __init__(self):
        self.models: dict[str, Model] = {
            # "YOLO11": YOLO(
            #     ""
            # ),
            "YOLO8": YOLO(
                "src/models/weights/yolov8m.pt"
            ),
            # "RT-DETR": RTDETR(
            #     ""
            # ),
            # "SAM": SAM("src/models/weights/sam2.1_b.pt"),
        }

    def predict(
        self, model_name: str, input_path: str, output_path: str | None
    ) -> Results:
        img = cv2.imread(input_path)
        if img is None:
            raise ValueError(f"Could not read image from {input_path}")

        model = self.models[model_name]

        if model_name == "SAM":
            pass
            # TODO

        return model.predict(input_path)[0]
