import json
from pathlib import Path


def normalize_coordinate(value: float, size: int) -> float:
    return round(value / size, 6)


def normalize_coordinates_as_str(
    x: float, y: float, width: int, height: int
) -> list[str]:
    norm_x = normalize_coordinate(x, width)
    norm_y = normalize_coordinate(y, height)
    return [str(norm_x), str(norm_y)]


def convert_Woodscape_to_Ultralytics(
    input_dir: Path, output_dir: Path, info_json_path: Path
):
    with open(info_json_path, "r", encoding="utf-8") as f:
        CLASSES = {name: idx for idx, name in enumerate(json.load(f)["classes"])}

    output_dir.mkdir(parents=True, exist_ok=True)

    for json_file in input_dir.glob("*.json"):
        with json_file.open("r", encoding="utf-8") as f:
            annotations = json.load(f)[json_file.name]

        height = annotations["image_height"]
        width = annotations["image_width"]

        label_file = output_dir / json_file.with_suffix(".txt").name
        lines = []
        for data in annotations["annotation"]:
            class_id = CLASSES.get(data["tags"][0], -1)
            if class_id == -1:
                continue

            segmentation = data["segmentation"]
            norm_coords = []
            for point in segmentation:
                norm_coords.extend(
                    normalize_coordinates_as_str(point[0], point[1], width, height)
                )

            lines.append(f"{class_id} " + " ".join(norm_coords))

        label_file.write_text("\n".join(lines))
