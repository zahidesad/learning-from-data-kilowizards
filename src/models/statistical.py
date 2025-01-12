import pickle
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
from .base import BaseModel


########################################################
# ARIMA
########################################################
class ArimaModel(BaseModel):
    """
    Wrapper for an ARIMA model using statsmodels.
    Inherits from BaseModel.
    """

    def __init__(self, name='ARIMA', order=(1, 0, 1)):
        super().__init__(name)
        self.order = order  # (p, d, q)

    def fit(self, X, y=None):
        """
        X is typically a time-indexed pd.Series (univariate).
        y is not used in classical ARIMA, but kept for compatibility.
        """
        self.model = ARIMA(X, order=self.order).fit()
        return self

    def predict(self, start=None, end=None, dynamic=False):
        if self.model is None:
            raise ValueError("Model is not fitted. Call fit() first.")
        return self.model.predict(start=start, end=end, dynamic=dynamic)

    def save(self, path: str):
        """
        Save the fitted ARIMA model using pickle.
        """
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)

    def load(self, path: str):
        """
        Load the ARIMA model from disk.
        """
        with open(path, 'rb') as f:
            self.model = pickle.load(f)


########################################################
# SARIMA
########################################################
class SarimaModel(BaseModel):
    """
    Wrapper for SARIMA (SARIMAX in statsmodels).
    Inherits from BaseModel.
    """

    def __init__(self, name='SARIMA', order=(1, 1, 1), seasonal_order=(0, 1, 1, 12)):
        super().__init__(name)
        self.order = order
        self.seasonal_order = seasonal_order

    def fit(self, X, y=None):
        """
        X is typically a univariate time series (pd.Series).
        """
        self.model = SARIMAX(X,
                             order=self.order,
                             seasonal_order=self.seasonal_order,
                             enforce_stationarity=False,
                             enforce_invertibility=False).fit()
        return self

    def predict(self, start=None, end=None, dynamic=False):
        if self.model is None:
            raise ValueError("Model is not fitted. Call fit() first.")
        return self.model.predict(start=start, end=end, dynamic=dynamic)

    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)

    def load(self, path: str):
        with open(path, 'rb') as f:
            self.model = pickle.load(f)


########################################################
# Prophet
########################################################
class ProphetModel(BaseModel):
    """
    Wrapper for Prophet.
    Expects DataFrame columns: ds (datetime) and y (target).
    """

    def __init__(self, name='Prophet', yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False):
        super().__init__(name)
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        # Prophet instance created in fit()

    def fit(self, X, y=None):
        """
        X must have columns: ds, y.
        """
        self.model = Prophet(
            yearly_seasonality=self.yearly_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            daily_seasonality=self.daily_seasonality
        )
        self.model.fit(X)  # Prophet is fit on a DataFrame
        return self

    def predict(self, X):
        """
        X must have column: ds (future or test datetimes).
        """
        if self.model is None:
            raise ValueError("Model is not fitted. Call fit() first.")
        forecast = self.model.predict(X)
        return forecast

    def save(self, path: str):
        """
        Save Prophet model using pickle (or model-specific serialization).
        """
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)

    def load(self, path: str):
        with open(path, 'rb') as f:
            self.model = pickle.load(f)
