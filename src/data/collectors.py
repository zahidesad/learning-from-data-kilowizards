# src/data/collectors.py

import os
import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, List


class EPIASCollector:
    """
    EPİAŞ verilerini toplamak için bir kolektör sınıfı.
    Yerel CSV veya EPİAŞ Şeffaflık API'lerinden veri alabilir.
    """

    def __init__(self, local_csv_path: Optional[str] = None, api_key: Optional[str] = None):
        self.local_csv_path = local_csv_path
        self.api_key = api_key
        self.base_url = "https://seffaflik.epias.com.tr/transparency/service"

    def load_local_data(self) -> pd.DataFrame:
        if self.local_csv_path is None or not os.path.exists(self.local_csv_path):
            raise FileNotFoundError(f"CSV dosyası bulunamadı: {self.local_csv_path}")
        df = pd.read_csv(self.local_csv_path, encoding='utf-8', delimiter=',')
        return df

    def fetch_api_data(
            self,
            start_date: str,
            end_date: str,
            market_type: str = "day-ahead"
    ) -> pd.DataFrame:
        if not self.api_key:
            raise ValueError("EPİAŞ API key belirtilmedi.")
        url = f"{self.base_url}/{market_type}?startDate={start_date}&endDate={end_date}&api_key={self.api_key}"
        resp = requests.get(url)
        if resp.status_code != 200:
            raise ConnectionError(f"API isteğinde hata: {resp.status_code} - {resp.text}")

        data_json = resp.json()
        records = data_json.get("body", {}).get("data", [])
        df = pd.DataFrame(records)
        return df

    def get_data(
            self,
            use_local: bool = True,
            start_date: Optional[str] = None,
            end_date: Optional[str] = None,
            market_type: str = "day-ahead"
    ) -> pd.DataFrame:
        if use_local:
            return self.load_local_data()
        else:
            if not start_date or not end_date:
                raise ValueError("start_date ve end_date belirtilmeli.")
            return self.fetch_api_data(start_date, end_date, market_type)


class WeatherBitCollector:
    """
    WeatherBit API'yi kullanarak alt saatlik hava durumu verisini (free tier 1 ay) çekmek için.
    """

    def __init__(self, api_key: str, lat: float, lon: float):
        self.api_key = api_key
        self.lat = lat
        self.lon = lon
        self.base_url = "https://api.weatherbit.io/v2.0/history/subhourly"

    def fetch_month_data(self, year: int, month: int) -> pd.DataFrame:
        # Ayın ilk ve son günü
        start_date = datetime(year, month, 1)
        if month == 12:
            end_date = datetime(year + 1, 1, 1) - timedelta(days=1)
        else:
            end_date = datetime(year, month + 1, 1) - timedelta(days=1)

        params = {
            "lat": self.lat,
            "lon": self.lon,
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d"),
            "key": self.api_key
        }
        resp = requests.get(self.base_url, params=params)
        if resp.status_code != 200:
            raise ConnectionError(f"WeatherBit isteğinde hata: {resp.status_code} - {resp.text}")

        data_json = resp.json()
        if "data" not in data_json:
            return pd.DataFrame()  # O ay verisi yoksa boş dön

        df = pd.DataFrame(data_json["data"])
        return df
