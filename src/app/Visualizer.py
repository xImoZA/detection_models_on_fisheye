import os
import subprocess

import cv2
import numpy as np
from ultralytics.engine.results import Results
from ultralytics.utils.plotting import colors


class Visualizer:
    def _visualize(self, predict: Results):
        visualized_img = predict.plot(conf=False, labels=False)

        annotation = self._create_annotation(predict, visualized_img.shape[0])

        final_img = np.hstack((visualized_img, annotation))

        return final_img

    def _create_annotation(self, result, img_height):
        classes = result.boxes.cls.unique()
        names = result.names

        annotation_width = 300
        annotation = np.ones((img_height, annotation_width, 3), dtype=np.uint8) * 255

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        line_height = 40
        start_y = 50

        for i, cls in enumerate(classes):
            class_id = int(cls)
            class_name = names[class_id]
            color = self._get_class_color(class_id)

            cv2.rectangle(
                annotation,
                (10, start_y + i * line_height - 20),
                (30, start_y + i * line_height),
                color,
                -1,
            )

            cv2.putText(
                annotation,
                class_name,
                (40, start_y + i * line_height - 5),
                font,
                font_scale,
                (0, 0, 0),
                thickness,
            )

        return annotation

    def _get_class_color(self, class_id):
        color = colors(int(class_id))  # RGB annotation
        return color[2], color[1], color[0]

    def save_and_visualize(self, output_path: str, model_name: str, predict: Results):
        final_img = self._visualize(predict)

        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        cv2.imwrite(output_path, final_img)

        self._open_image(output_path)

    def _open_image(self, image_path: str):
        try:
            subprocess.call(("xdg-open", image_path))
        except Exception as e:
            print(f"Не удалось открыть изображение: {e}")
