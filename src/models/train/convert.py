import json
import os

def normalize_coordinate(x: float, y: float, width: int, height: int) -> list[str]:
    return [str(round(x / width, 6)), str(round(y / height, 6))]

def convert_Woodscape_to_Ultralytics(input_dir: str, output_dir: str, info_json_path: str):
    with open(info_json_path, "r", encoding="utf-8") as f:
        CLASSES = {name: idx for idx, name in enumerate(json.load(f)["classes"])}

    os.makedirs(output_dir, exist_ok=True)

    for json_file in os.listdir(input_dir):
        with open(f"{input_dir}/{json_file}", "r", encoding="utf-8") as f:
            annotations = json.load(f)[json_file]

        height = annotations["image_height"]
        width = annotations["image_width"]

        label_file = os.path.join(output_dir, json_file.replace(".json", ".txt"))
        lines = []
        for data in annotations["annotation"]:
            class_id = CLASSES.get(data["tags"][0], -1)
            if class_id == -1:
                continue

            segmentation = data["segmentation"]
            norm_coords = []
            for point in segmentation:
                norm_coords.extend(normalize_coordinate(point[0], point[1], width, height))

            lines.append(f"{class_id} " + " ".join(norm_coords))

        with open(label_file, "w") as f:
            f.write("\n".join(lines))
