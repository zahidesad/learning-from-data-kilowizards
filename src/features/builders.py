from typing import Literal

import pandas as pd


def merge_datasets(elec_df: pd.DataFrame, weather_df: pd.DataFrame,
                   time_key_elec='Tarih', time_key_weather='timestamp_local',
                   how: Literal["left", "right", "inner", "outer", "cross"] = "left"):
    """
    Merges the electricity dataset with the weather dataset on a time key.

    Parameters:
    -----------
    elec_df : pd.DataFrame
        Electricity price DataFrame (with columns like: Tarih, Smf, Ptf, etc.)
    weather_df : pd.DataFrame
        Weather DataFrame (with columns like: timestamp_local, temp, wind_spd, etc.)
    time_key_elec : str
        The name of the datetime column in elec_df
    time_key_weather : str
        The name of the datetime column in weather_df
    how : str
        The type of merge (left, right, inner, outer)

    Returns:
    --------
    merged_df : pd.DataFrame
        The merged DataFrame containing electricity data and weather data
    """
    # Ensure both are datetime
    elec_df[time_key_elec] = pd.to_datetime(elec_df[time_key_elec], dayfirst=True)
    weather_df[time_key_weather] = pd.to_datetime(weather_df[time_key_weather])

    # Possibly round/truncate times if needed (e.g. weather data is hourly)
    # weather_df[time_key_weather] = weather_df[time_key_weather].dt.floor('H')

    # Merge
    merged_df = pd.merge(elec_df, weather_df,
                         left_on=time_key_elec,
                         right_on=time_key_weather,
                         how=how)
    return merged_df


def build_time_features(df: pd.DataFrame, datetime_col: str = 'Tarih'):
    """
    Creates time-based features like hour, day, day_of_week, month, etc.

    Parameters:
    -----------
    df : pd.DataFrame
    datetime_col : str

    Returns:
    --------
    df : pd.DataFrame
        The original df with new time-based features added
    """
    df[datetime_col] = pd.to_datetime(df[datetime_col])

    df['hour'] = df[datetime_col].dt.hour
    df['day'] = df[datetime_col].dt.day
    df['day_of_week'] = df[datetime_col].dt.dayofweek
    df['month'] = df[datetime_col].dt.month
    df['year'] = df[datetime_col].dt.year

    # Example: Is weekend?
    df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)

    return df


def build_rolling_features(df: pd.DataFrame,
                           target_col: str = 'Smf',
                           rolling_windows=(24, 48, 72),
                           groupby_col: str = None):
    """
    Adds rolling statistical features (mean, std, etc.) for the target column.

    Parameters:
    -----------
    df : pd.DataFrame
    target_col : str
        Column for which to create rolling features
    rolling_windows : list
        A list of window sizes (in *hours* or *rows*)
        e.g. [24, 48, 72] means 24h, 48h, 72h windows
    groupby_col : str
        If you have data from multiple regions/hours, you might group by something.
        If not needed, you can skip grouping.

    Returns:
    --------
    df : pd.DataFrame
        The df with rolling features appended
    """

    # Sort by time if needed to ensure correct rolling
    df = df.sort_values(by='Tarih')

    if groupby_col:
        group_object = df.groupby(groupby_col)
    else:
        # Just use the whole df
        group_object = [('', df)]

    feature_list = []
    for key, group in group_object:
        for window in rolling_windows:
            col_mean = f'{target_col}_rolling_mean_{window}'
            col_std = f'{target_col}_rolling_std_{window}'

            group[col_mean] = group[target_col].rolling(window=window, min_periods=1).mean()
            group[col_std] = group[target_col].rolling(window=window, min_periods=1).std()

        feature_list.append(group)

    df = pd.concat(feature_list).sort_index()

    return df


def build_lag_features_multi(df, cols_to_lag, lags=(1, 24), time_col='Tarih'):
    """
    Creates lag features for each column in cols_to_lag.
    For example, if cols_to_lag = ['Smf', 'Talepislemhacmi'] and lags = (1, 24),
    it will create:
      Smf_lag_1,  Smf_lag_24,
      Talepislemhacmi_lag_1, Talepislemhacmi_lag_24.
    """
    df = df.sort_values(by=time_col)

    for col in cols_to_lag:
        for lag in lags:
            df[f'{col}_lag_{lag}'] = df[col].shift(lag)

    # Optionally drop rows that have NaN from lagging
    df = df.dropna().reset_index(drop=True)
    return df