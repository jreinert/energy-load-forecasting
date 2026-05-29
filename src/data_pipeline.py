# -*- coding: utf-8 -*-
"""
data_pipeline.py
Fetches hourly electricity demand data from EIA and hourly temperature data from NOAA for the MISO region and saves raw CSVs to data/raw/
"""

# import libs
import os
import requests
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

# set api keys
EIA_API_KEY = os.getenv('EIA_API_KEY')
NOAA_TOKEN = os.getenv('NOAA_TOKEN')

# MISO region id in EIA api
EIA_REGION = 'MISO'

# NOAA station - Chicago O'Hare (GHCND:USW00094846)
# central to MISO footprint, reliable hrly data
NOAA_STATION = 'GHCND:USW00094846'

# function declarations
def fetch_eia_demand(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetches hourly electricity demand from EIA api v2 for MISO region
    
    Args:
        start_date: Format 'YYYY-MM-DD'
        end_date: Format 'YYYY-MM-DD'
    
    Returns: df with columns [datetime_utc, demand_mw]
    """
    print(f'fetching EIA demand data: {start_date} to {end_date}')

    url = 'https://api.eia.gov/v2/electricity/rto/region-data/data/'

    params = {'api_key': EIA_API_KEY,
              'frequency': 'hourly',
              'data[0]': 'value',
              'facets[respondent][]': EIA_REGION,
              'facets[type][]': 'D',
              'start': start_date + 'T00',
              'end': end_date + 'T23',
              'sort[0][column]': 'period',
              'sort[0][direction]': 'asc',
              'offset': 0,
              'length': 5000}
    
    all_records = []

    while True:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        records = data.get('response', {}).get('data', [])
        if not records:
            break

        all_records.extend(records)
        print(f'Fetched {len(all_records)} records so far...')

        # paginate if needed
        total = data.get('response', {}).get('total', 0)
        if len(all_records) >= int(total):
            break
        params['offset'] += 5000
    
    # check if all_records is empty and raise ValueError if true
    if not all_records:
        raise ValueError("No EIA data returned. Check API key and date range")
    
    # convert all_records to df, rename cols, convert data types, and sort values
    df = pd.DataFrame(all_records)
    df = df.rename(columns={'period': 'datetime_utc', 'value': 'demand_mw'})
    df['datetime_utc'] = pd.to_datetime(df['datetime_utc'])
    df['demand_mw'] = pd.to_numeric(df['demand_mw'], errors='coerce')
    df = df[['datetime_utc', 'demand_mw']].sort_values('datetime_utc').reset_index(drop=True)
    
    print(f'EIA data fetched: {len(df)} hourly records')
    # return df
    return df

def fetch_openmeteo_temperature(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch hourly temperature data from Open-Meteo API.
    Uses Chicago O'Hare coordinates — central to MISO footprint.
    No API key required.

    Args:
        start_date: Format 'YYYY-MM-DD'
        end_date:   Format 'YYYY-MM-DD'

    Returns:
        DataFrame with columns [datetime_utc, temp_c]
    """
    print(f"Fetching Open-Meteo temperature data: {start_date} to {end_date}")

    url = "https://archive.open-meteo.com/v1/archive"

    params = {
        "latitude": 41.9742,   # Chicago O'Hare
        "longitude": -87.9073,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "temperature_2m",
        "timezone": "UTC",
    }

    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json()

    timestamps = data["hourly"]["time"]
    temperatures = data["hourly"]["temperature_2m"]

    df = pd.DataFrame({
        "datetime_utc": pd.to_datetime(timestamps),
        "temp_c": temperatures
    })

    df["temp_c"] = pd.to_numeric(df["temp_c"], errors="coerce")
    df = df.sort_values("datetime_utc").reset_index(drop=True)

    print(f"  Open-Meteo data fetched: {len(df)} hourly records")
    return df

def fetch_noaa_temperature(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch hourly temperature data from NOAA CDO api

    Args:
        start_date: Format 'YYYY-MM-DD'
        end_date:   Format 'YYYY-MM-DD'

    Returns:
        DataFrame with columns [datetime_utc, temp_c]
    """
    print(f'Fetching NOAA temperature data: {start_date} to {end_date}')

    url = 'https://www.ncdc.noaa.gov/cdo-web/api/v2/data'
    headers = {'token': NOAA_TOKEN}

    all_records = []
    offset = 1
    limit = 1000
    
    # noaa cdo has 1 yr max per req; chunk by yr
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    chunks = _chunk_date_range(start, end, days=365)

    for chunk_start, chunk_end in chunks:
        offset = 1
        while True:
            params = {'datasetid': 'LCD',
                      'stationid': NOAA_STATION,
                      'datatypeid': 'HourlyDryBulbTemperature',
                      'startdate': chunk_start.strftime('%Y-%m-%d'),
                      'enddate': chunk_end.strftime('%Y-%m-%d'),
                      'units': 'metric',
                      'limit': limit,
                      'offset': offset}
            
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            data = response.json()

            records = data.get('results', [])
            
            # check if records is empty and break if True
            if not records:
                break

            all_records.extend(records)
            print(f'Fetched {len(all_records)} temperature records so far...')

            # check if len of records is less than the limit and break if True
            if len(records) < limit:
                break
            offset += limit

    # check if all_records is empty and break if True
    if not all_records:
        raise ValueError('No NOAA data returned. Check token and station ID')
    
    # convert to df, rename cols, convert data types, and sort values
    df = pd.DataFrame(all_records)
    df = df.rename(columns={'date': 'datetime_utc', 'value': 'temp_c'})
    df['datetime_utc'] = pd.to_datetime(df['datetime_utc'])
    df['temp_c'] = pd.to_numeric(df['temp_c'], errors='coerce')
    df = df[['dattime_utc', 'temp_c']].sort_values('datetime_utc').reset_index(drop=True)

    # drop dups & keep first
    df = df.drop_duplicates(subset='datetime_utc').reset_index(drop=True)

    print(f'NOAA data fetched: {len(df)} hourly records')
    # return df
    return df

def _chunk_date_range(start: datetime, end: datetime, days: int):
    """
    Split date range into chunks of max days length
    """
    chunks = []
    current = start

    while current < end:
        chunk_end = min(current + timedelta(days=days - 1), end)
        chunks.append((current, chunk_end))
        current = chunk_end + timedelta(days=1)

    # return chunks
    return chunks

def save_raw_data(df: pd.DataFrame, filename: str) -> None:
    """
    Save a df to data/raw/
    """
    path = os.path.join('data', 'raw', filename)
    df.to_csv(path, index=False) # write out as csv
    return pd.read_csv(path, parse_dates=['datetime_utc'])

def load_raw_data(filename: str) -> pd.DataFrame:
    """
    Loads df from data/raw/
    """
    path = os.path.join('data', 'raw', filename)
    return pd.read_csv(path, parse_dates=['datetime_utc'])

def load_weather_data(filepath: str) -> pd.DataFrame:
    """
    Load and standardize IEM weather data from CSV.
    Resamples to hourly, converts F to C, handles missing values.

    Args:
        filepath: Path to the raw IEM CSV file

    Returns:
        DataFrame with columns [datetime_utc, temp_c]
    """
    print(f'Loading weather data from {filepath}')

    df = pd.read_csv(filepath)

    # Replace missing value markers with NaN
    df = df.replace('M', pd.NA)
    df = df.replace('T', pd.NA)

    # Parse timestamps and convert temperature
    df['datetime_utc'] = pd.to_datetime(df['valid'], utc=True)
    df['temp_c'] = (pd.to_numeric(df['tmpf'], errors='coerce') - 32) * 5 / 9

    df = df[['datetime_utc', 'temp_c']].sort_values('datetime_utc').reset_index(drop=True)

    # Resample to hourly — take the mean of any observations within each hour
    df = df.set_index('datetime_utc')
    df = df.resample('h').mean()
    df = df.interpolate(method='time')  # fill any gaps
    df = df.reset_index()

    # Strip timezone info to match EIA format
    df['datetime_utc'] = df['datetime_utc'].dt.tz_localize(None)

    print(f'  Weather data loaded: {len(df)} hourly records')
    return df

def run_pipeline(start_date: str, end_date: str) -> None:
    """
    Full pipeline run - fetch EIA and NOAA data and save to data/raw

    Args:
        start_date: Format 'YYYY-MM-DD'
        end_date:   Format 'YYYY-MM-DD'
    """
    # fetch and save EIA demand
    eia_path = os.path.join('data', 'raw', 'eia_demand_miso.csv')
    if os.path.exists(eia_path):
        print('EIA data already exists — skipping fetch.')
        eia_df = load_raw_data('eia_demand_miso.csv')
    else:
        eia_df = fetch_eia_demand(start_date, end_date)
        save_raw_data(eia_df, 'eia_demand_miso.csv')

    # Load weather from local file
    weather_df = load_weather_data(
        os.path.join('data', 'raw', 'weather_temperature_raw.csv')
    )
    save_raw_data(weather_df, 'weather_temperature.csv')

    # fetch and save NOAA temp
    #noaa_df = fetch_noaa_temperature(start_date, end_date)
    #weather_df = fetch_openmeteo_temperature(start_date, end_date)
    #save_raw_data(weather_df, 'noaa_temperature.csv')

    print('\nPipeline complete')
    print(f'EIA records: {len(eia_df)}')
    print(f'Weather records: {len(weather_df)}')

if __name__ == '__main__':
    # pull 3 yrs of data 
    run_pipeline(start_date='2021-01-01', end_date='2023-12-31')

