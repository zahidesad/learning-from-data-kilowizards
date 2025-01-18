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

    def __init__(self, name='ARIMA', order=(1, 1, 1)):
        """
        Initialize the ARIMA model with a specific order.

        Parameters:
        - name: str, optional
            Name of the model (default: 'ARIMA').
        - order: tuple
            The (p, d, q) parameters for ARIMA.
        """
        super().__init__(name)
        self.order = order
        self.model = None

    def fit(self, X, y=None):
        """
        Fit the ARIMA model to the given time series.

        Parameters:
        - X: pd.Series
            The univariate time series data for training.
        """
        self.model = ARIMA(X, order=self.order).fit()
        return self

    def forecast(self, steps):
        """
        Forecast future values using the ARIMA model.

        Parameters:
        - steps: int
            Number of steps to forecast ahead.

        Returns:
        - pd.Series
            Forecasted values.
        """
        if self.model is None:
            raise ValueError("The model must be fitted before forecasting.")
        return self.model.forecast(steps=steps)

    def predict(self, steps=None, start=None, end=None, dynamic=False):
        """
        Predict values using the ARIMA model.

        Parameters:
        - steps: int, optional
            Number of steps to forecast ahead.
        - start, end: int, optional
            Range for prediction if forecasting over an interval.
        - dynamic: bool, optional
            Dynamic forecasting flag.

        Returns:
        - pd.Series
            Predicted values.
        """
        if self.model is None:
            raise ValueError("Model is not fitted. Call fit() first.")

        if steps is not None:
            return self.model.forecast(steps=steps)
        else:
            return self.model.predict(start=start, end=end, dynamic=dynamic)

    def save(self, path: str):
        """Save the fitted ARIMA model using pickle."""
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)

    def load(self, path: str):
        """Load the ARIMA model from a file."""
        with open(path, 'rb') as f:
            self.model = pickle.load(f)


class SarimaModel(BaseModel):
    """
    Wrapper for a SARIMA model using statsmodels.
    Inherits from BaseModel.
    """

    def __init__(self, name='SARIMA', order=(1, 1, 1), seasonal_order=(1, 1, 0, 12)):
        """
        Initialize the SARIMA model with specific orders.

        Parameters:
        - name: str, optional
            Name of the model (default: 'SARIMA').
        - order: tuple
            The (p, d, q) order for ARIMA.
        - seasonal_order: tuple
            The (P, D, Q, s) order for SARIMA's seasonal component.
        """
        super().__init__(name)
        self.order = order
        self.seasonal_order = seasonal_order
        self.model = None

    def fit(self, X, y=None):
        """
        Fit the SARIMA model to the given time series.

        Parameters:
        - X: pd.Series
            The univariate time series data for training.
        """
        self.model = SARIMAX(X,
                             order=self.order,
                             seasonal_order=self.seasonal_order,
                             enforce_stationarity=False,
                             enforce_invertibility=False).fit()
        return self

    def forecast(self, steps):
        """
        Forecast future values using the SARIMA model.

        Parameters:
        - steps: int
            Number of steps to forecast ahead.

        Returns:
        - pd.Series
            Forecasted values.
        """
        if self.model is None:
            raise ValueError("The model must be fitted before forecasting.")
        return self.model.forecast(steps=steps)

    def predict(self, steps=None, start=None, end=None, dynamic=False):
        """
        Predict values using the SARIMA model.

        Parameters:
        - steps: int, optional
            Number of steps to forecast ahead.
        - start, end: int, optional
            Range for prediction if forecasting over an interval.
        - dynamic: bool, optional
            Dynamic forecasting flag.

        Returns:
        - pd.Series
            Predicted values.
        """
        if self.model is None:
            raise ValueError("Model is not fitted. Call fit() first.")

        if steps is not None:
            return self.forecast(steps=steps)
        else:
            return self.model.predict(start=start, end=end, dynamic=dynamic)

    def save(self, path: str):
        """Save the fitted SARIMA model using pickle."""
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)

    def load(self, path: str):
        """Load the SARIMA model from a file."""
        with open(path, 'rb') as f:
            self.model = pickle.load(f)


class ProphetModel(BaseModel):
    """
    Wrapper for a Prophet model.
    """

    def __init__(self, name='Prophet', yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False):
        """
        Initialize the Prophet model.

        Parameters:
        - name: str, optional
            Name of the model (default: 'Prophet').
        - yearly_seasonality, weekly_seasonality, daily_seasonality: bool, optional
            Flags to enable/disable specific seasonal components.
        """
        super().__init__(name)
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        self.model = Prophet(
            yearly_seasonality=self.yearly_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            daily_seasonality=self.daily_seasonality
        )

    def fit(self, X, y=None):
        """
        Fit the Prophet model to the given DataFrame.

        Parameters:
        - X: pd.DataFrame
            DataFrame with 'ds' (dates) and 'y' (values).
        """
        self.model = Prophet(
            yearly_seasonality=self.yearly_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            daily_seasonality=self.daily_seasonality
        )
        self.model.fit(X)
        return self

    def forecast(self, steps, freq='D'):
        """
        Forecast future values using the Prophet model.

        Parameters:
        - steps: int
            Number of steps to forecast ahead.
        - freq: str, optional
            Frequency for future DataFrame creation.

        Returns:
        - pd.DataFrame
            Forecasted values including 'yhat' (predictions).
        """
        if self.model is None:
            raise ValueError("The model must be fitted before forecasting.")
        future = self.model.make_future_dataframe(periods=steps, freq=freq)
        return self.model.predict(future)

    def add_regressor(self, name):
        """
        Add a regressor to the Prophet model.

        Parameters:
        - name: str
            The name of the regressor column in the input data.
        """
        self.model.add_regressor(name)

    def predict(self, steps=None, freq='D'):
        """
        Alias for the forecast method for compatibility.
        """
        return self.forecast(steps=steps, freq=freq)

    def save(self, path: str):
        """Save the Prophet model using pickle."""
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)

    @classmethod
    def load(cls, path: str):
        """Load a Prophet model from a file."""
        with open(path, 'rb') as f:
            model = pickle.load(f)

        instance = cls()
        instance.model = model
        return instance
