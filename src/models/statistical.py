import pickle
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
from .base import BaseModel

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

    def predict(self, steps=None, start=None, end=None, dynamic=False):
        if self.model is None:
            raise ValueError("Model is not fitted. Call fit() first.")

        if steps is not None:
            return self.model.forecast(steps=steps)  # Tahmin modu
        else:
            return self.model.predict(start=start, end=end, dynamic=dynamic)  # Tahmin aralığı

    def forecast(self, steps):
        # Check if the model is fitted before forecasting
        if self.model is None:
            raise ValueError("The model must be fitted before forecasting.")
        return self.model.forecast(steps=steps)

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


class SarimaModel(BaseModel):
    """
    Wrapper for SARIMA (SARIMAX in statsmodels).
    Inherits from BaseModel.
    """

    def __init__(self, name='SARIMA', order=(1, 1, 1), seasonal_order=(0, 1, 1, 12)):
        """
        Initialize SARIMA model with order and seasonal_order parameters.

        Parameters:
        - name: str, optional (default="SARIMA")
            Name of the model.
        - order: tuple, optional (default=(1, 1, 1))
            The (p, d, q) order of the model for the number of AR parameters,
            differences, and MA parameters.
        - seasonal_order: tuple, optional (default=(0, 1, 1, 12))
            The (P, D, Q, s) order of the seasonal components of the model.
        """
        super().__init__(name)
        self.order = order  # (p, d, q)
        self.seasonal_order = seasonal_order  # (P, D, Q, s)
        self.model = None

    def fit(self, X, y=None):
        """
        Fit the SARIMA model to the provided time series data.

        Parameters:
        - X: pd.Series
            Time-indexed univariate time series to fit the model.
        - y: None
            y is not utilized in SARIMA but kept for compatibility.

        Returns:
        - self: SarimaModel
            The fitted model object.
        """
        self.model = SARIMAX(X,
                             order=self.order,
                             seasonal_order=self.seasonal_order,
                             enforce_stationarity=False,
                             enforce_invertibility=False).fit()
        return self

    def predict(self, start=None, end=None, steps=None, dynamic=False):
        """
        Predict future values using the fitted SARIMA model.

        Parameters:
        - start: int, optional
            Start of the prediction interval.
        - end: int, optional
            End of the prediction interval.
        - steps: int, optional
            Number of steps to forecast ahead.
        - dynamic: bool, optional (default=False)
            If True, uses dynamic forecasting.

        Returns:
        - pd.Series
            Predicted values for the specified range.
        """
        if self.model is None:
            raise ValueError("Model is not fitted. Call fit() first.")

        # If steps is provided, calculate `end` based on steps
        if steps is not None:
            if start is None:
                start = self.model.data.shape[0]
            end = start + steps - 1

        return self.model.predict(start=start, end=end, dynamic=dynamic)

    def forecast(self, steps):
        """
        Forecast future values for a specified number of steps.

        Parameters:
        - steps: int
            The number of future time steps to forecast.

        Returns:
        - np.ndarray
            Forecasted values for the specified steps.
        """
        if self.model is None:
            raise ValueError("The model must be fitted before forecasting.")
        return self.model.forecast(steps=steps)

    def save(self, path: str):
        """
        Save the fitted SARIMA model to a file using pickle.

        Parameters:
        - path: str
            The file path to save the model.
        """
        if self.model is None:
            raise ValueError("No model to save. Train a model before saving.")
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)

    def load(self, path: str):
        """
        Load a SARIMA model from a pickle file.

        Parameters:
        - path: str
            The filepath to load the model from.
        """
        with open(path, 'rb') as f:
            self.model = pickle.load(f)


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
        self.model = None  # Prophet instance

    def fit(self, X, y=None):
        """
        Fit the Prophet model to the provided DataFrame.

        Parameters:
        - X: pd.DataFrame
            Must have columns 'ds' (datetimes) and 'y' (values).
        """
        self.model = Prophet(
            yearly_seasonality=self.yearly_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            daily_seasonality=self.daily_seasonality
        )
        self.model.fit(X)  # Prophet expects a DataFrame
        return self

    def predict(self, X=None, steps=None, freq='D'):
        """
        Predict future values or on a provided DataFrame.

        Parameters:
        - X: pd.DataFrame, optional
            DataFrame with a 'ds' column for test datetimes.
        - steps: int, optional
            Number of time steps to forecast ahead.
        - freq: str, optional (default='D')
            Frequency of the forecasted dates (e.g., 'D' for daily).

        Returns:
        - pd.DataFrame
            Forecasted values, including the predicted mean and components.
        """
        if self.model is None:
            raise ValueError("Model is not fitted. Call fit() first.")

        if steps is not None:  # Eğer `steps` belirtilmişse
            future = self.model.make_future_dataframe(periods=steps, freq=freq)
        elif X is not None:  # Eğer manuel bir DataFrame verilmişse
            if 'ds' not in X.columns:
                raise ValueError("DataFrame must contain a 'ds' column for datetimes.")
            future = X
        else:
            raise ValueError("Either X or steps must be provided.")

        return self.model.predict(future)

    def save(self, path: str):
        """
        Save Prophet model using pickle (or model-specific serialization).

        Parameters:
        - path: str
            File path to save the model.
        """
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)

    @classmethod
    def load(cls, path: str):
        """
        Load a Prophet model from a pickle file.

        Parameters:
        - path: str
            File path to load the model.

        Returns:
        - cls: ProphetModel
            A new instance of the ProphetModel with the loaded model.
        """
        with open(path, 'rb') as f:
            # Prophet modeli aç ve yeni bir ProphetModel örneği oluştur
            model = pickle.load(f)

        # Yeni ProphetModel örneği oluştur
        instance = cls()
        instance.model = model
        return instance
