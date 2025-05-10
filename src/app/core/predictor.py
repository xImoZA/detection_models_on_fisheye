from typing import Any

from numpy.typing import NDArray
from ultralytics.engine.results import Results

from src.app.utils.constants import MODEL_TYPE


class ModelPredictor:
    def __init__(self, model: MODEL_TYPE, track: bool):
        self.model: MODEL_TYPE = model
        self.track: bool = track

        if track and not hasattr(model, "track"):
            raise ValueError(f"Model {type(model).__name__} does not support tracking")

    def predict(self, img: NDArray[Any]) -> Results:
        if self.track:
            return self.model.track(img, persist=True)[0]
        return self.model.predict(img)[0]
