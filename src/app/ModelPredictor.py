import numpy as np
from load_model import MODEL_TYPE
from numpy.typing import NDArray
from ultralytics.engine.results import Results


class ModelPredictor:
    def __init__(self, model: MODEL_TYPE):
        self.model: MODEL_TYPE = model

    def predict(self, img: NDArray[np.uint8]) -> Results:
        return self.model.predict(img)[0]
