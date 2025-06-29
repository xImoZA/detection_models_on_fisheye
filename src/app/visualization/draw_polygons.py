from pathlib import Path

import cv2 as cv
import numpy as np
from numpy.typing import NDArray

from src.app.utils.constants import GT_COLOR


def draw_polygons_from_file(
    image: NDArray[np.uint8], gt_file: Path, color: tuple[int, int, int] = GT_COLOR
) -> NDArray[np.uint8]:
    img_height, img_width = image.shape[:2]

    thickness = 2

    try:
        with gt_file.open("r", encoding="utf-8") as f:
            for polygon in f.readlines():
                parts = list(map(float, polygon.strip().split()))
                coords = np.array(parts[1:]).reshape(-1, 2)
                polygon_abs = (coords * [img_width, img_height]).astype(np.int32)

                cv.polylines(
                    image,
                    [polygon_abs],
                    isClosed=True,
                    color=color,
                    thickness=thickness,
                )
    except Exception as e:
        raise ValueError(f"Error processing ground truth file {gt_file}: {str(e)}") from e

    return image
