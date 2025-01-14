import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def correlation_selector(df: pd.DataFrame, target_col: str = 'Smf', threshold: float = 0.1):
    """
    Select features whose absolute correlation with target is above threshold.

    Parameters:
    -----------
    df : pd.DataFrame
    target_col : str
    threshold : float

    Returns:
    --------
    selected_features : list
        List of columns passing the correlation threshold
    """
    corr_vals = df.corr()[target_col].drop(target_col)

    # Choose features above threshold
    selected_features = corr_vals[abs(corr_vals) >= threshold].index.tolist()

    print(f"Features correlated with {target_col} above {threshold}: {selected_features}")
    return selected_features


def embedded_rf_selector(df: pd.DataFrame, target_col: str,
                         candidate_features: list,
                         n_estimators=100, random_state=42):
    """
    Use RandomForestRegressor's feature_importances_ to select top features.

    Parameters:
    -----------
    df : pd.DataFrame
    target_col : str
    candidate_features : list
        The set of features to consider
    n_estimators : int
    random_state : int

    Returns:
    --------
    feature_importances : pd.Series
        Sorted feature importances from the random forest
    """
    X = df[candidate_features].fillna(0)
    y = df[target_col].fillna(0)

    model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
    model.fit(X, y)

    importances = model.feature_importances_
    feature_importances = pd.Series(importances, index=candidate_features)
    feature_importances = feature_importances.sort_values(ascending=False)

    return feature_importances


def xgb_feature_importance_selector(df: pd.DataFrame, target_col: str, test_size: float = 0.2, random_state: int = 42):
    """
    A function to preprocess data and calculate feature importances using XGBoost.

    Parameters:
    -----------
    df : pd.DataFrame
        The input DataFrame with features and target.
    target_col : str
        The name of the target column for prediction.
    test_size : float, default=0.2
        The proportion of the dataset to include in the test split.
    random_state : int, default=42
        The random state to use for splitting and model reproducibility.

    Returns:
    --------
    importance_df : pd.DataFrame
        A DataFrame containing features and their importance scores,
        sorted by importance in descending order.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]

    for col in X.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])

    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.fillna(X.mean(), inplace=True)
    y.fillna(y.mean(), inplace=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    model = XGBRegressor(enable_categorical=True, random_state=random_state)
    model.fit(X_train, y_train)

    importances = model.feature_importances_
    importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)

    return importance_df