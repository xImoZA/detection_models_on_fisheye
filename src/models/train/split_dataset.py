from sklearn.model_selection import train_test_split
import os
from shutil import copyfile

def copy_files(file_list, source_dir, target_dir, extension=".png"):
    os.makedirs(target_dir, exist_ok=True)
    
    for file_name in file_list:
        source_path = os.path.join(source_dir, file_name.replace(".png", extension))
        target_path = os.path.join(target_dir, file_name.replace(".png", extension))
        
        if os.path.exists(source_path):
            copyfile(source_path, target_path)
        else:
            raise ValueError(f"Файл {source_path} не найден.")

def split_dataset(dataset: str, annotations_path: str, output_dir: str):
    if not os.path.exists(dataset):
        raise FileNotFoundError(f"Input dir not found: {dataset}")

    train, _X = train_test_split(os.listdir(dataset), test_size=0.3, shuffle=True, random_state=42)
    val, test = train_test_split(_X, test_size=0.5, shuffle=True, random_state=42)

    copy_files(train, dataset, f"{output_dir}/images/train")
    copy_files(train, annotations_path, f"{output_dir}/labels/train", extension=".txt")

    copy_files(val, dataset, f"{output_dir}/images/val")
    copy_files(val, annotations_path, f"{output_dir}/labels/val", extension=".txt")

    copy_files(test, dataset, f"{output_dir}/images/test")
    copy_files(test, annotations_path, f"{output_dir}/labels/test", extension=".txt")
      
