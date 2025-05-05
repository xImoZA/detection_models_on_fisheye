import numpy as np
from load_model import MODEL_TYPE
from ultralytics.engine.results import Results


class ModelPredictor:
    def __init__(self, model: MODEL_TYPE):
        self.model: MODEL_TYPE = model

    def predict(self, img: np.ndarray) -> Results:
        return self.model.predict(img)[0]
