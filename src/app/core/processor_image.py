from pathlib import Path

import cv2 as cv

from src.app.core.predictor import ModelPredictor
from src.app.visualization.visualizer import Visualizer


def determine_output_path(input_path: Path, model_name: str, output_path: Path | None = None) -> Path:
    base_name = f"{input_path.stem}_{model_name}"
    suffix = input_path.suffix

    if output_path:
        parent = output_path
    else:
        parent = input_path.parent

    return parent / f"{base_name}{suffix}"


class ProcessorImage:
    def __init__(self, output_path: Path | None, model_name: str, predictor: ModelPredictor, visualizer: Visualizer):
        self.output_path = output_path
        self.model_name = model_name
        self.predictor = predictor
        self.visualizer = visualizer

    def process_single_image(self, input_path: Path, ground_truth: Path | None) -> None:
        output_path = determine_output_path(input_path, self.model_name, self.output_path)

        img = cv.imread(str(input_path))
        if img is None:
            raise ValueError(f"Could not read image from {input_path}")

        prediction = self.predictor.predict(img)
        self.visualizer.save_and_visualize(output_path, prediction, ground_truth)
