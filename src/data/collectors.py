import os
import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, List


class EPIASCollector:
    """
    A collector class for retrieving EPİAŞ data.
    It can fetch data from a local CSV file or EPİAŞ Transparency APIs.
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
    A class for retrieving sub-hourly weather data using the WeatherBit API
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

    def fetch_year_data(self, year: int, month: int) -> pd.DataFrame:
        # Ayın ilk ve son günü
        start_date = datetime(year, month, 1)
        end_date = datetime(year + 1, month, 1) - timedelta(days=1)

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

    def fetch_data_in_chunks(self, start_date, end_date, freq='1Y'):
        """
        Veriyi parça parça çeker ve birleştirir.

        Parameters:
        -----------
        fetch_function : function
            Belirli tarih aralığında veri çeken işlev. (örn. API sorgusu yapan)
        start_date : str or datetime
            Başlangıç tarihi (format: 'YYYY-MM-DD').
        end_date : str or datetime
            Bitiş tarihi (format: 'YYYY-MM-DD').
        freq : str
            Çekim parçalarının boyutları ('1Y' -> yıllık, '1M' -> aylık).

        Returns:
        --------
        pd.DataFrame
            Çekilen ve birleştirilmiş veri.
        """
        # Tarihleri datetime objelerine dönüştür
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)

        # Başlangıç ve bitiş tarihleri arasında parçalar üret
        try:
            df = pd.read_csv("weather_data_test.csv")
        except:
            df = pd.DataFrame()
        chunk_start = start_date

        while chunk_start < end_date:
            chunk_end = min(chunk_start + timedelta(days=30), end_date)

            # Fetch (API'den veri çekme)
            print(f"Fetching data from {chunk_start.date()} to {chunk_end.date()}...")
            chunk_data = self.fetch_function(chunk_start, chunk_end)

            # Eğer veri boş değilse listeye ekle
            if df is None:
                df = chunk_data
            elif chunk_data is not None and not chunk_data.empty:
                merged_df = pd.concat([df, chunk_data], ignore_index=True)
                # save as csv
                merged_df.to_csv(f"weather_data_test.csv", index=False)
                df = merged_df


            # Bir sonraki parçaya git
            chunk_start = chunk_end + timedelta(days=1)

        # Tüm parçaları birleştir
        if df is not None:
            return df
        else:
            return pd.DataFrame()  # Eğer hiçbir veri yoksa boş bir DataFrame döndür

    def fetch_function(self, start_date, end_date):
        """
        Örneksel veri çekme işlevi. Gerçek çağrılar için API entegrasyonu yapılır.
        """
        params = {
            "lat": self.lat,
            "lon": self.lon,
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d"),
            "key": self.api_key
        }

        # Proxy tanımlama
        proxies = {
            "http": "http://username:password@proxyaddress:port",
            "https": "http://username:password@proxyaddress:port"
        } # Proxy URL varsa ekler

        # HTTP isteği yapma
        try:
            resp = requests.get(self.base_url, params=params, proxies=proxies, timeout=10)
            if resp.status_code != 200:
                raise ConnectionError(f"WeatherBit isteğinde hata: {resp.status_code} - {resp.text}")

            data_json = resp.json()
            if "data" not in data_json:
                return pd.DataFrame()  # O ay verisi yoksa boş dön

            df = pd.DataFrame(data_json["data"])
            return df

        except requests.exceptions.ProxyError:
            raise ConnectionError("Proxy sunucusuna bağlanılamadı. Lütfen proxy ayarlarını kontrol edin.")
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"HTTP isteği sırasında bir hata meydana geldi: {str(e)}")