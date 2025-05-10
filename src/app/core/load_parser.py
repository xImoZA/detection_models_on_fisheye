import argparse
from pathlib import Path


def load_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Object detection visualization tool with model predictions")
    parser.add_argument(
        "-i",
        "--input_path",
        type=Path,
        required=True,
        help="Path to the input image file",
    )
    parser.add_argument(
        "-o",
        "--output_path",
        type=Path,
        help="Path to save the output image (default: input with model suffix)",
    )
    parser.add_argument(
        "-gt",
        "--ground_truth",
        type=Path,
        help="Path to the ground truth file in Ultralytics format (.txt file with the same name as the image)",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="YOLOv8",
        choices=["YOLOv8", "YOLOv11", "RT-DETR", "SAM"],
        help="Detection model to use (default: YOLOv8)",
    )
    parser.add_argument(
        "-t",
        "--track",
        action="store_true",
        help="Enable object tracking (only for supported models like YOLOv8)",
    )
    parser.add_argument(
        "-b",
        "--boxes",
        action="store_true",
        help="Enable bounding boxes visualization",
    )
    parser.add_argument(
        "-l",
        "--labels",
        action="store_true",
        help="Enable class labels visualization",
    )

    return parser.parse_args()
