from pathlib import Path

import cv2 as cv
import numpy as np
from ultralytics.engine.results import Results
from ultralytics.utils.plotting import colors


class Visualizer:
    def _visualize(self, prediction: Results):
        visualized_img = prediction.plot(conf=False, labels=False)

        annotation = self._create_annotation(prediction, visualized_img.shape[0])

        final_img = np.hstack((visualized_img, annotation))

        return final_img

    def _create_annotation(self, result, img_height):
        classes = result.boxes.cls.unique()
        names = result.names

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

        return annotation

    @staticmethod
    def _get_class_color(class_id):
        color = colors(int(class_id))  # RGB annotation
        return color[2], color[1], color[0]

    def save_and_visualize(self, output_path: Path, prediction: Results):
        final_img = self._visualize(prediction)

        output_path.parent.mkdir(parents=True, exist_ok=True)

        cv.imwrite(str(output_path), final_img)

        cv.imshow("Visualization", final_img)
        cv.waitKey(0)
        cv.destroyAllWindows()
