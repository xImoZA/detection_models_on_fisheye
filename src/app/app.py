import argparse
import os

from src.app.ModelPredict import ModelPredict
from src.app.Visualizer import Visualizer


def main():
    parser = argparse.ArgumentParser(
        description="Visualize object detection predictions with annotation"
    )
    parser.add_argument("input_path", help="Path to the input image")
    parser.add_argument(
        "output_path",
        nargs="?",
        default=None,
        help="Optional path to save the output image",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="YOLO8",
        choices=["YOLO11", "YOLO8", "RT-DETR", "SAM"],
        help="Model to use for detection",
    )

    args = parser.parse_args()

    if not os.path.exists(args.input_path):
        raise FileNotFoundError(f"Input file not found: {args.input_path}")

    model_predict = ModelPredict()
    visualizer = Visualizer()

    pred = model_predict.predict(args.model, args.input_path, args.output_path)

    if args.output_path is None:
        base, ext = os.path.splitext(args.input_path)
    else:
        base, ext = os.path.splitext(args.output_path)

    ext = ext if ext else ".jpg"
    output_path = f"{base}_{args.model}{ext}"

    visualizer.save_and_visualize(output_path, args.model, pred)


if __name__ == "__main__":
    main()
