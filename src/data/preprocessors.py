import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Any
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder


class DataPreprocessor:
    """Handle data preprocessing tasks."""

    def __init__(self):
        # Birden fazla scaler veya encoder saklamak isterseniz dictionary kullanabilirsiniz
        self.scalers: Dict[str, Any] = {}
        self.encoders: Dict[str, Any] = {}

    def parse_dates(self, df: pd.DataFrame, date_col: str, date_format: Optional[str] = None) -> pd.DataFrame:
        """
        Tarih kolonunu datetime tipine çevirir.
        :param df: Giriş veri seti
        :param date_col: Tarih bilgisinin yer aldığı kolon adı
        :param date_format: Opsiyonel format (ör. '%d/%m/%Y %H:%M')
        """
        if date_format:
            df[date_col] = pd.to_datetime(df[date_col], format=date_format, errors='coerce')
        else:
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        return df

    def handle_missing_values(
        self,
        df: pd.DataFrame,
        method: str = 'interpolate_linear',
        numeric_cols: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Eksik değer doldurma yöntemi:
          - interpolate_linear
          - mean
          - median
          - ffill / bfill
        """
        if numeric_cols is None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        if method == 'interpolate_linear':
            df[numeric_cols] = df[numeric_cols].interpolate(method='linear')
        elif method == 'mean':
            for col in numeric_cols:
                df[col].fillna(df[col].mean(), inplace=True)
        elif method == 'median':
            for col in numeric_cols:
                df[col].fillna(df[col].median(), inplace=True)
        elif method == 'ffill':
            df[numeric_cols] = df[numeric_cols].fillna(method='ffill')
        elif method == 'bfill':
            df[numeric_cols] = df[numeric_cols].fillna(method='bfill')
        return df

    def remove_outliers(self, df: pd.DataFrame, columns: List[str], method: str = 'iqr') -> pd.DataFrame:
        """
        Belirli kolonlardaki aykırı değerleri (outlier) belirli bir yöntemle ele alır:
        - 'iqr': (Q1 - 1.5*IQR, Q3 + 1.5*IQR) dışında kalan değerleri çıkarır
        - 'zscore': Z-skoru |z| < 3 olarak filtreler
        """
        if method == 'iqr':
            for col in columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
        elif method == 'zscore':
            from scipy import stats
            for col in columns:
                df = df[(np.abs(stats.zscore(df[col])) < 3)]
        return df

    def encode_categorical(
        self,
        df: pd.DataFrame,
        columns: List[str],
        method: str = 'label'
    ) -> pd.DataFrame:
        """
        Kategorik kolonları kodlama:
        - method='label' -> LabelEncoder
        - method='onehot' -> get_dummies
        """
        if method == 'label':
            for col in columns:
                if col not in self.encoders:
                    le = LabelEncoder()
                    df[col] = df[col].astype(str)  # NaN veya numeric karışık ise str'ye çevir
                    df[col] = le.fit_transform(df[col])
                    self.encoders[col] = le
                else:
                    # Daha önce eğitilmiş bir encoder varsa
                    le = self.encoders[col]
                    df[col] = df[col].astype(str)
                    df[col] = le.transform(df[col])
        elif method == 'onehot':
            df = pd.get_dummies(df, columns=columns, drop_first=True)
        return df

    def scale_features(
        self,
        df: pd.DataFrame,
        columns: List[str],
        method: str = 'standard'
    ) -> pd.DataFrame:
        """
        Özellik ölçekleme (scaling):
        - 'standard': StandardScaler
        - 'minmax': MinMaxScaler
        - 'robust': RobustScaler
        """
        if method == 'standard':
            scaler = StandardScaler()
            df[columns] = scaler.fit_transform(df[columns])
            self.scalers['standard'] = scaler
        elif method == 'minmax':
            scaler = MinMaxScaler()
            df[columns] = scaler.fit_transform(df[columns])
            self.scalers['minmax'] = scaler
        elif method == 'robust':
            scaler = RobustScaler()
            df[columns] = scaler.fit_transform(df[columns])
            self.scalers['robust'] = scaler
        else:
            raise ValueError(f"Geçersiz ölçeklendirme metodu: {method}")
        return df

    def create_time_features(self, df: pd.DataFrame, date_col: str) -> pd.DataFrame:
        """
        Zaman serisi analizinde kullanılabilecek ek özellikler üretir:
        hour, day, weekday, month, year ...
        """
        if not np.issubdtype(df[date_col].dtype, np.datetime64):
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')

        df['hour'] = df[date_col].dt.hour
        df['day'] = df[date_col].dt.day
        df['weekday'] = df[date_col].dt.weekday
        df['month'] = df[date_col].dt.month
        df['year'] = df[date_col].dt.year
        return df

    def create_rolling_features(
        self,
        df: pd.DataFrame,
        target_col: str,
        window_sizes: List[int] = [24, 72]
    ) -> pd.DataFrame:
        """
        Belirli pencerelerde rolling mean, std vb. istatistikleri oluşturur.
        """
        for w in window_sizes:
            df[f"{target_col}_rolling_mean_{w}"] = df[target_col].rolling(window=w).mean()
            df[f"{target_col}_rolling_std_{w}"] = df[target_col].rolling(window=w).std()
        return df

    def create_lag_features(
        self,
        df: pd.DataFrame,
        target_col: str,
        lags: List[int] = [1, 24]
    ) -> pd.DataFrame:
        """
        Gecikmeli (lag) özellikler ekler. Örn. 1 saat önceki değer, 24 saat önceki değer vb.
        """
        for lag in lags:
            df[f"{target_col}_lag_{lag}"] = df[target_col].shift(lag)
        return df

    def remove_duplicates(self, df: pd.DataFrame, column: str = 'timestamp_local') -> pd.DataFrame:
        """
        Removes duplicate rows based on the specified column. Keeps the first occurrence of the duplicate value.

        Parameters:
            df (pd.DataFrame): The input DataFrame.
            column (str): The column to inspect for duplicate values. Default is 'timestamp_local'.

        Returns:
            pd.DataFrame: The DataFrame with duplicates removed.
        """
        # Drop duplicate rows based on the specified column
        return df.drop_duplicates(subset=[column], keep='first')

    def aggregate_to_hourly(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Veri setini saatlik veriye dönüştürür.
        timestamp_utc kolonuna göre gruplama yapar.
        Ortalamalarını alır, sayısal olmayan kolonların (ör. weather) ilk değerini alır.
        Kolonların orijinal sırası korunur.

        :param df: Giriş veri seti
        :return: Saatlik olarak birleştirilmiş ve kolon sırası korunmuş veri seti
        """
        # Orijinal kolon sırasını kaydet
        original_columns = df.columns.tolist()

        # timestamp_utc ve timestamp_local'i datetime türüne çevir
        df['timestamp_utc'] = pd.to_datetime(df['timestamp_utc'])
        df['timestamp_local'] = pd.to_datetime(df['timestamp_local'])

        # timestamp_utc'yi saatlik binlere yuvarla
        df['hourly_timestamp_utc'] = df['timestamp_utc'].dt.floor('h')
        df['hourly_timestamp_local'] = df['timestamp_local'].dt.floor('h')

        # Sayısal ve sayısal olmayan kolonları ayır
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        non_numeric_columns = df.select_dtypes(exclude=[np.number, 'datetime']).columns.tolist()

        # Aggregation: Sayısal kolonlar için ortalama, diğerleri için ilk değer
        aggregated_df = df.groupby('hourly_timestamp_utc').agg({**{col: 'mean' for col in numeric_columns},
                                                                **{col: 'first' for col in non_numeric_columns},
                                                                'timestamp_local': 'first'}).reset_index()

        # İsimleri güncelle
        aggregated_df = aggregated_df.rename(columns={'hourly_timestamp_utc': 'timestamp_utc',
                                                      'hourly_timestamp_local': 'timestamp_local'})

        # Kolon sırasını orijinal sıraya göre düzenle
        aggregated_df = aggregated_df[original_columns]

        return aggregated_df

def preprocess_dataframe(
    df: pd.DataFrame,
    date_col: Optional[str] = None,
    missing_method: str = 'mean',
    outlier_cols: Optional[List[str]] = None,
    outlier_method: str = 'iqr'
) -> pd.DataFrame:
    """
    Minimal wrapper that uses DataPreprocessor to do basic steps:
    - parse dates (optional)
    - handle missing values (default 'mean')
    - remove outliers (optional)
    Modify or extend as you like.
    """
    dp = DataPreprocessor()

    # 1) parse date column if provided
    if date_col:
        df = dp.parse_dates(df, date_col=date_col)

    # 2) handle missing values
    df = dp.handle_missing_values(df, method=missing_method)

    # 3) remove outliers if columns are specified
    if outlier_cols:
        df = dp.remove_outliers(df, columns=outlier_cols, method=outlier_method)

    # you can add more calls, like encode_categorical, scale_features, etc.
    return df

