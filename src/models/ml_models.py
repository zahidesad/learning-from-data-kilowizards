# ml_models.py

import pickle
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None

from .base import BaseModel


########################################################
# RANDOM FOREST
########################################################
class RandomForestModel(BaseModel):
    """
    Wrapper for a RandomForestRegressor.
    """

    def __init__(self, name='RandomForest', n_estimators=100, max_depth=None, random_state=42):
        super().__init__(name)
        # Instantiate the underlying regressor
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state
        )

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        if self.model is None:
            raise ValueError("Model is not fitted. Call fit() first.")
        return self.model.predict(X)

    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)

    def load(self, path: str):
        with open(path, 'rb') as f:
            self.model = pickle.load(f)


########################################################
# SVR
########################################################
class SVRModel(BaseModel):
    """
    Wrapper for Support Vector Regressor.
    """

    def __init__(self, name='SVR', kernel='rbf', C=1.0, epsilon=0.1):
        super().__init__(name)
        self.model = SVR(kernel=kernel, C=C, epsilon=epsilon)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        if self.model is None:
            raise ValueError("Model is not fitted. Call fit() first.")
        return self.model.predict(X)

    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)

    def load(self, path: str):
        with open(path, 'rb') as f:
            self.model = pickle.load(f)


########################################################
# XGBOOST
########################################################
class XGBModel(BaseModel):
    """
    Wrapper for an XGBRegressor.
    """

    def __init__(self, name='XGBoost', n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42):
        super().__init__(name)
        if XGBRegressor is None:
            raise ImportError("XGBoost is not installed. Please install xgboost.")
        self.model = XGBRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=random_state
        )

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        if self.model is None:
            raise ValueError("Model is not fitted. Call fit() first.")
        return self.model.predict(X)

    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)

    def load(self, path: str):
        with open(path, 'rb') as f:
            self.model = pickle.load(f)
