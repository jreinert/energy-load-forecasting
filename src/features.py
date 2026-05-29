# -*- coding: utf-8 -*-
"""
features.py
Builds the feature matrix from raw EIA demand and weather data.
All feature engineering logic lives here.
"""
# import libs
import os
import numpy as np
import pandas as pd
import holidays

from data_pipeline import load_raw_data, save_raw_data

# function declarations
def merge_demand_and_weather(demand_df: pd.DataFrame,
                             weather_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge EIA demand and weather data on datetime_utc.
    Uses an inner join — only keeps timestamps present in both datasets.
    """
    # inner join on dfs
    df = pd.merge(demand_df, weather_df, on='datetime_utc', how='inner')
    df = df.sort_values('datetime_utc').reset_index(drop=True)

    print(f'Merged: dataset: {len(df)} hourly records')

    # return df
    return df

def add_cyclical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode time-based features as sine/cosine pairs.

    Why sine/cosine? Raw integers (hour 0-23) imply hour 23 and hour 0
    are far apart. Cyclical encoding wraps these onto a circle so the
    model understands hour 23 and hour 0 are adjacent.

    Each cycle needs BOTH sin and cos to uniquely identify a position.
    Sin alone can't distinguish between the rising and falling sides
    of the cycle.
    """
    # hr of day - 24hr cycle
    df['hour_sin'] = np.sin(2 * np.pi * df['datetime_utc'].dt.hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['datetime_utc'].dt.hour / 24)

    # day of wk - 7 day cycle (0 = Mon, 6 = Sun)
    df['dow_sin'] = np.sin(2 * np.pi * df['datetime_utc'].dt.dayofweek / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['datetime_utc'].dt.dayofweek / 7)

    # mnth - 12 mnth cycle
    df['month_sin'] = np.sin(2 * np.pi * df['datetime_utc'].dt.dayofweek / 7)
    df['month_cos'] = np.cos(2 * np.pi * df['datetime_utc'].dt.dayofweek / 7)

    # return df
    print('Added cyclical date/time features')
    return df

def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add lagged demand values as features.

    Why lags? Electricity demand is autocorrelated — demand right now
    is strongly correlated with demand at the same time yesterday,
    two days ago, and last week. We're giving the model explicit
    access to this historical pattern.

    Important: shift(n) shifts values DOWN by n rows, meaning each row
    gets the value from n hours earlier. This is correct — we want
    each row to see what demand looked like n hours in the past,
    not the future.
    """
    df['lag_24'] = df['demand_mw'].shift(24)
    df['lag_48'] = df['demand_mw'].shift(48)
    df['lag_168'] = df['demand_mw'].shift(168)
    
    # return df
    print('Added lag features')
    return df

def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add rolling mean features.

    Why rolling means? They capture recent demand trends and smooth
    out short-term noise. The 24-hour window captures daily rhythm,
    the 168-hour window captures weekly rhythm.

    min_periods=1 means we start computing as soon as we have at
    least 1 value rather than waiting for the full window to fill.
    This avoids NaNs at the start of the dataset.
    """
    df['rolling_mean_24'] = (df['demand_mw']
                             .shift(1) # prevents data leakage by using only past values
                             .rolling(window=24,min_periods=1)
                             .mean())
    
    df['rolling_mean_168'] = (df['demand_mw']
                              .shift(1)
                              .rolling(window=168, min_periods=1)
                              .mean())
    
    # return df
    print('Added rolling mean features')
    return df

def add_holiday_flag(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a binary flag for US federal holidays based off datetime_utc col
    """
    us_holidays = holidays.US(years=df['datetime_utc'].dt.year.unique())

    df['is_holiday'] = (df['datetime_utc']
                        .dt.date
                        .astype(str)
                        .map(lambda d: 1 if d in us_holidays else 0))
    
    # return df
    print('Added holiday flag')
    return df

def drop_na_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop rows with NaN values that were introduced by lag & rolling feature additions
    """
    before = len(df)
    df = df.dropna().reset_index(drop=True)
    after = len(df)
    
    # return df
    print(f'Dropped {before-after} rows with NaN values ({after} rows remaining)')
    return df

def build_feature_matrix() -> pd.DataFrame:
    """
    Full pipeline that loads raw data, engineers features, and saves processed data
    Returns: DataFrame with all features ready for modeling
    """
    print('Building feature matrix...')
    
    # load raw data
    demand_df = load_raw_data('eia_demand_miso.csv')
    weather_df = load_raw_data('weather_temperature.csv')

    # build features step x step
    df = merge_demand_and_weather(demand_df, weather_df)
    df = add_cyclical_features(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_holiday_flag(df)
    df = drop_na_rows(df)

    # save processed data
    output_path = os.path.join('data', 'processed', 'features.csv')
    df.to_csv(output_path, index=False)

    print(f'\nFeature Matrix saved: {output_path}')
    print(f'Shape: {df.shape}')
    print(f'Cols: {list(df.columns)}')

    # return df
    return df

if __name__ == '__main__':
    build_feature_matrix()