import cv2 as cv
import numpy as np
from numpy.typing import NDArray
from ultralytics.engine.results import Results
from ultralytics.utils.plotting import colors

from src.app.utils.constants import GT_COLOR


def create_annotation(prediction: Results, img_height: int) -> NDArray[np.uint8]:
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

        rgb_color = colors(int(class_id))  # RGB annotation
        bgr_color = rgb_color[2], rgb_color[1], rgb_color[0]  # BGR annotation

        cv.rectangle(
            annotation,
            (start_x_rectangle, start_y + i * line_height - height_rectangle),
            (end_x_rectangle, start_y + i * line_height),
            bgr_color,
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
