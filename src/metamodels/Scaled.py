import numpy as np
from sklearn.preprocessing import StandardScaler


@DeprecationWarning
class Scaled:

    def __init__(self, model):
        self.model = model
        self.scaler = StandardScaler()

    def fit(self, X, y):
        self.scaler: StandardScaler = StandardScaler().fit(X)
        training_scaled = self.scaler.transform(X)
        self.model.fit(training_scaled, y)
        return self

    def predict(self, X) -> np.ndarray:
        input_scaled = self.scaler.transform(X)
        return self.model.predict(input_scaled)

    def predict_proba(self, X) -> np.ndarray:
        input_scaled = self.scaler.transform(X)
        return self.model.predict_proba(input_scaled)