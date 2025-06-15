from pathlib import Path

import cv2 as cv
import numpy as np
from numpy.typing import NDArray
from ultralytics.engine.results import Results

from src.app.visualization.create_annotation import create_annotation
from src.app.visualization.draw_polygons import draw_polygons_from_file


class Visualizer:
    def __init__(self, boxes: bool, labels: bool):
        self.boxes = boxes
        self.labels = labels

    def _visualize(self, prediction: Results, gt_path: Path | None) -> NDArray[np.uint8]:
        img = prediction.plot(conf=False, labels=self.labels, boxes=self.boxes)

        if gt_path is not None:
            img = draw_polygons_from_file(img, gt_path)

        annotation = create_annotation(prediction, img.shape[0])

        final_img = np.hstack((img, annotation))

        return final_img

    def save_and_visualize(self, output_path: Path, prediction: Results, gt_path: Path | None) -> None:
        final_img = self._visualize(prediction, gt_path)

        output_path.parent.mkdir(parents=True, exist_ok=True)

        cv.imwrite(str(output_path), final_img)

        cv.imshow("Detected objects", final_img)
        cv.waitKey(0)
        cv.destroyAllWindows()
