import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def evaluate_regression(y_true, y_pred, prefix=""):
    """
    Evaluates a regression model's predictions using common metrics.

    Parameters
    ----------
    y_true : array-like
        Ground truth (correct) target values.
    y_pred : array-like
        Estimated target values returned by the model.
    prefix : str, optional
        A string prefix to prepend to each returned metric key.
        Useful if you're evaluating multiple sets of predictions in one dictionary.

    Returns
    -------
    dict
        Dictionary containing the following keys:
        - f"{prefix}MAE": Mean Absolute Error
        - f"{prefix}MSE": Mean Squared Error
        - f"{prefix}RMSE": Root Mean Squared Error
        - f"{prefix}R2": R^2 coefficient of determination


    {'MAE': 0.09999999999999987, 'MSE': 0.006666666666666616,
     'RMSE': 0.0816496580927726, 'R2': 0.999...}
    """
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    return {
        f"{prefix}MAE": mae,
        f"{prefix}MSE": mse,
        f"{prefix}RMSE": rmse,
        f"{prefix}R2": r2
    }


def print_regression_metrics(y_true, y_pred, label=""):
    """
    Prints MAE, MSE, RMSE, and R2 for a given set of predictions.

    Parameters
    ----------
    y_true : array-like
        Ground truth values.
    y_pred : array-like
        Model predictions.
    label : str, optional
        A label to print before the metrics.

    MyModel Regression Metrics:
      MAE : 0.1
      MSE : ...
      RMSE: ...
      R^2 : ...
    """
    results = evaluate_regression(y_true, y_pred)
    if label:
        print(f"{label} Regression Metrics:")
    else:
        print("Regression Metrics:")

    print(f"  MAE : {results['MAE']:.4f}")
    print(f"  MSE : {results['MSE']:.4f}")
    print(f"  RMSE: {results['RMSE']:.4f}")
    print(f"  R^2 : {results['R2']:.4f}")
