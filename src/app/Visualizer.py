from pathlib import Path

import cv2 as cv
import numpy as np
from draw_ultralytics_polygons_from_file import (
    GT_COLOR,
    draw_ultralytics_polygons_from_file,
)
from numpy.typing import NDArray
from ultralytics.engine.results import Results
from ultralytics.utils.plotting import colors


class Visualizer:
    def _visualize(self, prediction: Results, gt_path: Path | None) -> NDArray[np.uint8]:
        has_masks = hasattr(prediction, "masks") and prediction.masks is not None
        img = prediction.plot(conf=False, labels=False, boxes=not has_masks)

        if gt_path is not None:
            img = draw_ultralytics_polygons_from_file(img, gt_path)

        annotation = self._create_annotation(prediction, img.shape[0])

        final_img = np.hstack((img, annotation))

        return final_img

    def _create_annotation(self, prediction: Results, img_height: int) -> NDArray[np.uint8]:
        classes = prediction.boxes.cls.unique()
        names = prediction.names

        annotation_width = 500
        annotation = np.ones((img_height, annotation_width, 3), dtype=np.uint8) * 255

        start_y = 50

        start_x_rectangle = 10
        end_x_rectangle = 30
        height_rectangle = 20
        thickness_rectangle = -1

        start_x = 40
        height_text = 10
        font = cv.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        text_color = (0, 0, 0)
        thickness_front = 2
        line_height = 40

        for i, cls in enumerate(classes):
            class_id = int(cls)
            class_name = names[class_id]
            color = Visualizer._get_class_color(class_id)

            cv.rectangle(
                annotation,
                (start_x_rectangle, start_y + i * line_height - height_rectangle),
                (end_x_rectangle, start_y + i * line_height),
                color,
                thickness_rectangle,
            )

            cv.putText(
                annotation,
                class_name,
                (start_x, start_y + i * line_height - height_text),
                font,
                font_scale,
                text_color,
                thickness_front,
            )

        cv.rectangle(
            annotation,
            (
                start_x_rectangle,
                start_y + len(classes) * line_height - height_rectangle,
            ),
            (end_x_rectangle, start_y + len(classes) * line_height),
            GT_COLOR,
            thickness_rectangle,
        )

        cv.putText(
            annotation,
            "ground truth",
            (start_x, start_y + len(classes) * line_height - height_text),
            font,
            font_scale,
            text_color,
            thickness_front,
        )
        return annotation

    @staticmethod
    def _get_class_color(class_id: int) -> tuple[int, int, int]:
        color = colors(int(class_id))  # RGB annotation
        return color[2], color[1], color[0]

    def save_and_visualize(self, output_path: Path, prediction: Results, gt_path: Path | None) -> None:
        final_img = self._visualize(prediction, gt_path)

        output_path.parent.mkdir(parents=True, exist_ok=True)

        cv.imwrite(str(output_path), final_img)

        cv.imshow("Detected objects", final_img)
        cv.waitKey(0)
        cv.destroyAllWindows()
