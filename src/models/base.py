from abc import ABC, abstractmethod

class BaseModel(ABC):
    """Base class for all forecasting models."""

    def __init__(self, name: str):
        self.name = name
        self.model = None

    @abstractmethod
    def fit(self, X, y):
        """Train the model."""
        pass

    @abstractmethod
    def predict(self, X):
        """Make predictions."""
        pass

    @abstractmethod
    def save(self, path: str):
        """Save model to disk."""
        pass

    @abstractmethod
    def load(self, path: str):
        """Load model from disk."""
        pass
