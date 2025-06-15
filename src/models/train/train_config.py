from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DatasetConfig:
    config_path: str

    def __post_init__(self) -> None:
        if not Path(self.config_path).exists():
            raise FileNotFoundError(f"Dataset config file not found: {self.config_path}")


@dataclass
class ModelConfig:
    pretrained_weights: str
    output_dir: str

    def __post_init__(self) -> None:
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)


@dataclass
class TrainingConfig:
    epochs: int
    batch_size: int
    image_size: int
    device: str
    workers: int
    save_period: int

    def __post_init__(self) -> None:
        if self.device not in ["cuda", "cpu"]:
            raise ValueError("Device must be either 'cuda' or 'cpu'")
        if self.image_size <= 0:
            raise ValueError("Image size must be positive")


@dataclass
class OptimizerConfig:
    name: str
    lr0: float
    warmup_epochs: int

    def __post_init__(self) -> None:
        if self.name not in ["AdamW", "SGD", "Adam", "RMSprop"]:
            raise ValueError("Unsupported optimizer")
        if self.lr0 <= 0:
            raise ValueError("Learning rate must be positive")


@dataclass
class ExperimentConfig:
    name: str


class YamlConfig:
    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self._raw_config = self._load_config()

        self.dataset = DatasetConfig(**self._raw_config["dataset"])
        self.model = ModelConfig(**self._raw_config["model"])
        self.training = TrainingConfig(**self._raw_config["training"])
        self.optimizer = OptimizerConfig(**self._raw_config["optimizer"])
        self.experiment = ExperimentConfig(**self._raw_config["experiment"])

    def _load_config(self) -> dict[str, Any]:
        with self.config_path.open("r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
            if not isinstance(config, dict):
                raise ValueError(
                    f"Config file {self.config_path} must contain a dictionary, got {type(config).__name__} instead"
                )
            return config

    def get_path_pretrained_weights(self) -> str:
        return str(self.model.pretrained_weights)

    def get_train_params(self) -> dict[str, Any]:
        return {
            "data": self.dataset.config_path,
            "epochs": self.training.epochs,
            "batch": self.training.batch_size,
            "imgsz": self.training.image_size,
            "device": self.training.device,
            "workers": self.training.workers,
            "save_period": self.training.save_period,
            "project": self.model.output_dir,
            "name": self.experiment.name,
            "optimizer": self.optimizer.name,
            "lr0": self.optimizer.lr0,
            "warmup_epochs": self.optimizer.warmup_epochs,
        }
