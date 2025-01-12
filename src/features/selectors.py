import pandas as pd
from sklearn.ensemble import RandomForestRegressor


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