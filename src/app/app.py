import argparse
from pathlib import Path

import cv2 as cv
from load_model import load_model
from ModelPredictor import ModelPredictor
from Visualizer import Visualizer


def determine_output_path(input_path: Path, model_name: str, output_path: Path | None = None) -> Path:
    if output_path:
        if output_path.is_dir():
            return output_path / f"{input_path.stem}_{model_name}{input_path.suffix}"
        return output_path

    return input_path.parent / f"{input_path.stem}_{model_name}{input_path.suffix}"


def main() -> None:
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
        help="Path to the ground truth file in Ultralytics format",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="YOLOv8",
        choices=["YOLOv8", "YOLOv11", "RT-DETR", "SAM"],
        help="Detection model to use (default: YOLOv8)",
    )

    args = parser.parse_args()

    if not args.input_path.exists():
        raise FileNotFoundError(f"Input file not found: {args.input_path}")

    if not args.ground_truth.exists():
        raise FileNotFoundError(f"Ground truth file not found: {args.ground_truth}")

    output_path = determine_output_path(args.input_path, args.model, args.output_path)

    try:
        model = load_model(args.model)
        model_predictor = ModelPredictor(model)
        visualizer = Visualizer()

        img = cv.imread(str(args.input_path))
        if img is None:
            raise ValueError(f"Could not read image from {args.input_path}")

        prediction = model_predictor.predict(img)
        visualizer.save_and_visualize(output_path, prediction, args.ground_truth)

    except Exception as e:
        print(f"Error processing image: {e}")
        raise


if __name__ == "__main__":
    main()
