import numpy as np
from ultralytics.engine.results import Results

from load_model import MODEL_TYPE


class ModelPredictor:
    def __init__(self, model: MODEL_TYPE):
        self.model: MODEL_TYPE = model

    def predict(self, img: np.ndarray) -> Results:
        return self.model.predict(img)[0]
