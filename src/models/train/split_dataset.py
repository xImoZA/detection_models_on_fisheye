from pathlib import Path
from shutil import copyfile

from sklearn.model_selection import train_test_split


def copy_files(file_list: list[str], source_dir: Path, target_dir: Path, extension: str = ".png") -> None:
    target_dir.mkdir(parents=True, exist_ok=True)

    for file_name in file_list:
        stem = Path(file_name).stem
        source_path = source_dir / f"{stem}{extension}"
        target_path = target_dir / f"{stem}{extension}"

        copyfile(source_path, target_path)


def split_dataset(dataset: Path, annotations_path: Path, output_dir: Path) -> None:
    if not dataset.exists():
        raise FileNotFoundError(f"Input dir not found: {dataset}")

    file_names = [f.name for f in dataset.iterdir() if f.is_file()]

    train, _X = train_test_split(file_names, test_size=0.3, shuffle=True, random_state=42)
    val, test = train_test_split(_X, test_size=0.5, shuffle=True, random_state=42)

    train_img_dir = output_dir / "images" / "train"
    train_lbl_dir = output_dir / "labels" / "train"
    val_img_dir = output_dir / "images" / "val"
    val_lbl_dir = output_dir / "labels" / "val"
    test_img_dir = output_dir / "images" / "test"
    test_lbl_dir = output_dir / "labels" / "test"

    copy_files(train, dataset, train_img_dir)
    copy_files(train, annotations_path, train_lbl_dir, extension=".txt")

    copy_files(val, dataset, val_img_dir)
    copy_files(val, annotations_path, val_lbl_dir, extension=".txt")

    copy_files(test, dataset, test_img_dir)
    copy_files(test, annotations_path, test_lbl_dir, extension=".txt")
