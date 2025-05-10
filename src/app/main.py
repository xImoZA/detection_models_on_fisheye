from pathlib import Path

from src.app.core.load_model import load_model
from src.app.core.load_parser import load_parser
from src.app.core.predictor import ModelPredictor
from src.app.core.processor_image import ProcessorImage
from src.app.visualization.visualizer import Visualizer


def main() -> None:
    args = load_parser()

    input_path = args.input_path
    output_path = args.output_path
    ground_truth = args.ground_truth
    model_name = args.model
    track = args.track
    boxes = args.boxes
    labels = args.labels

    if not input_path.exists():
        raise FileNotFoundError(f"Input path not found: {args.input_path}")

    if ground_truth is not None and not ground_truth.exists():
        raise FileNotFoundError(f"Ground truth path not found: {args.ground_truth}")

    if output_path is not None:
        if output_path.exists() and not output_path.is_dir():
            output_path = output_path.parent
        output_path.mkdir(parents=True, exist_ok=True)

    try:
        model = load_model(model_name)
        model_predictor = ModelPredictor(model, track)
        visualizer = Visualizer(boxes, labels)
        processor = ProcessorImage(output_path, model_name, model_predictor, visualizer)

        if args.input_path.is_dir():
            _process_directory(args.input_path, args.ground_truth, processor)
        else:
            processor.process_single_image(args.input_path, args.ground_truth)

    except Exception as e:
        print(f"Error processing image: {e}")
        raise


def _process_directory(input_dir: Path, gt_dir: Path | None, processor: ProcessorImage) -> None:
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

    all_files = sorted(
        [p for p in input_dir.rglob("*") if p.is_file() and p.suffix.lower() in image_extensions],
        key=lambda x: str(x).lower(),
    )

    for img in all_files:
        if img.is_file() and img.suffix.lower() in image_extensions:
            gt_path = None
            if gt_dir:
                rel_path = img.relative_to(input_dir)
                gt_path = gt_dir / f"{rel_path.with_suffix('.txt')}"

                if not gt_path.exists():
                    print(f"Ground truth not found for {img.name}")
                    gt_path = None

            processor.process_single_image(img, gt_path)


if __name__ == "__main__":
    main()
