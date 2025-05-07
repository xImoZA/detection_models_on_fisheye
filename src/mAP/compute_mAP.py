import json

import numpy as np

WOODSCAPE_WIDTH = 1280
WOODSCAPE_HEIGHT = 966


def compute_iou(box1: list[float], box2: list[float]) -> float:
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    xi_min = max(x1_min, x2_min)
    yi_min = max(y1_min, y2_min)
    xi_max = min(x1_max, x2_max)
    yi_max = min(y1_max, y2_max)
    inter_area = max(0.0, xi_max - xi_min) * max(0.0, yi_max - yi_min)

    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area


def get_annotation(path_annotation: str, file_name: str) -> dict[str, list[list[float]]]:
    objects: dict[str, list[list[float]]] = {
        "vehicles": [],
        "bicycle": [],
        "person": [],
    }

    with open(f"{path_annotation}/{file_name}.txt", encoding="utf-8") as annotation_file:
        for string in annotation_file.readlines():
            coordinates = string.split(",")
            cls_name = coordinates[0]
            annotation_coord = [
                float(coordinates[2]),
                float(coordinates[3]),
                float(coordinates[4]),
                float(coordinates[5]),
            ]

            if cls_name in objects:
                objects[cls_name].append(annotation_coord)

    return objects


def analyze_detection_results(
    path_result: str, path_annotation: str, iou_threshold: float, normalize: bool
) -> tuple[dict[str, list[tuple[float, str]]], dict[str, int]]:
    tp_and_fp: dict[str, list[tuple[float, str]]] = {
        "vehicles": [],
        "person": [],
        "bicycle": [],
    }
    false_negative = {"vehicles": 0, "person": 0, "bicycle": 0}

    with open(path_result, encoding="utf-8") as file:
        data = json.load(file)

        for component in data:
            filename = component["filename"].split("/")[-1][:-4]

            objects = get_annotation(path_annotation, filename)

            for expected_obj in component["objects"]:
                if normalize:
                    detecting_coord = normalize_coordinate(
                        expected_obj["relative_coordinates"]["center_x"],
                        expected_obj["relative_coordinates"]["center_y"],
                        expected_obj["relative_coordinates"]["width"],
                        expected_obj["relative_coordinates"]["height"],
                    )
                else:
                    detecting_coord = [
                        expected_obj["relative_coordinates"]["x_min"],
                        expected_obj["relative_coordinates"]["y_min"],
                        expected_obj["relative_coordinates"]["x_max"],
                        expected_obj["relative_coordinates"]["y_max"],
                    ]

                cls_name = expected_obj["name"]
                if cls_name in [
                    "car",
                    "truck",
                    "bus",
                    "motorbike",
                    "motorcycle",
                    "train",
                    "boat",
                    "airplane",
                    "aeroplane",
                ]:
                    cls_name = "vehicles"
                elif cls_name in ["bicycle", "person"]:
                    pass
                else:
                    continue

                matched = False
                max_iou = 0.0
                best_match = None
                for annotation_coordinate in objects[cls_name]:
                    iou = compute_iou(detecting_coord, annotation_coordinate)
                    if iou > max_iou and iou >= iou_threshold:
                        matched = True
                        max_iou = iou
                        best_match = annotation_coordinate

                if matched and best_match is not None:
                    tp_and_fp[cls_name].append((expected_obj["confidence"], "true_positive"))
                    objects[cls_name].remove(best_match)
                else:
                    tp_and_fp[cls_name].append((expected_obj["confidence"], "false_positive"))

            for cls in objects:
                false_negative[cls] += len(objects[cls])

        return tp_and_fp, false_negative


def compute_voc_ap(precision: list[float], recall: list[float]) -> float:
    recall_levels = np.linspace(0.0, 1.0, 11)

    interpolated_precision = np.zeros_like(recall_levels)
    for i, recall_level in enumerate(recall_levels):
        relevant_precisions = [p for r, p in zip(recall, precision) if r >= recall_level]
        if relevant_precisions:
            interpolated_precision[i] = max(relevant_precisions)
        else:
            interpolated_precision[i] = 0.0

    ap = np.mean(interpolated_precision)
    return float(ap)


def compute_map(tp_and_fp: dict[str, list[tuple[float, str]]], fn: dict[str, int]) -> float:
    APs: float = 0.0
    for key in tp_and_fp.keys():
        true_positive: int = 0
        false_positive: int = 0
        false_negative: int = fn[key]

        precision = []
        recall = []

        sorted_list = sorted(tp_and_fp[key], key=lambda item: item[0], reverse=True)
        for obj in sorted_list:
            if obj[1] == "true_positive":
                true_positive += 1
            else:
                false_positive += 1

            precision.append(true_positive / (true_positive + false_positive))
            recall.append(true_positive / (true_positive + false_negative))

        class_ap = compute_voc_ap(precision, recall)

        APs += class_ap

    return APs / len(tp_and_fp.keys())


def normalize_coordinate(center_x: float, center_y: float, width: float, height: float) -> list[float]:
    x_min = (center_x - width / 2) * WOODSCAPE_WIDTH
    y_min = (center_y - height / 2) * WOODSCAPE_HEIGHT
    x_max = (center_x + width / 2) * WOODSCAPE_WIDTH
    y_max = (center_y + height / 2) * WOODSCAPE_HEIGHT

    return [x_min, y_min, x_max, y_max]


def main(
    path_result: str,
    path_annotation: str,
    iou_threshold: float = 0.5,
    normalize: bool = False,
) -> float:
    tp_and_fp, false_negative = analyze_detection_results(path_result, path_annotation, iou_threshold, normalize)
    return compute_map(tp_and_fp, false_negative)


if __name__ == "__main__":
    print(main("result.json", "/home/sashka/Загрузки/box_2d_annotations/"))
