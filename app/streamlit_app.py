# -*- coding: utf-8 -*-
"""
streamlit_app.py
Interactive web application for MISO Energy Demand Forecasting.
Powered by a PyTorch LSTM trained on EIA hourly demand data.

Run with:
    streamlit run app/streamlit_app.py
"""

# import libraries
import os
import sys
import numpy as np
import pandas as pd
import torch
import joblib
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Add src to path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from model import EnergyLSTM
from dataset import FEATURE_COLS, TARGET_COL, SEQUENCE_LENGTH

# page config
st.set_page_config(page_title='MISO Energy Demand Forecasting',
                   page_icon='⚡',
                   layout='wide',
                   initial_sidebar_state='expanded')

# function declarations
# cached loaders
@st.cache_resource
def load_model():
    """
    Load trained LSTM Model

    @st.cache_resources caches the model object across all sessions
    """
    # create object of the model
    model = EnergyLSTM(input_size=13,
                       hidden_size=128,
                       num_layers=2,
                       dropout=0.2,
                       output_size=1)
    
    # load the best model
    model_path = os.path.join(os.path.dirname(__file__),'..', 'models', 'best_model.pt')

    model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))

    # evaluate the model
    model.eval()

    return model

@st.cache_resource
def load_scalers():
    """
    Load fitted scalers saved during training
    Cached so they're only loaded once
    """
    base = os.path.join(os.path.dirname(__file__), '..', 'models')
    feature_scaler = joblib.load(os.path.join(base, 'feature_scaler.pkl'))
    target_scaler = joblib.load(os.path.join(base, 'target_scaler.pkl'))

    return feature_scaler, target_scaler

@st.cache_data
def load_data():
    """
    Load the full feature matrix and cache the df
    """
    path = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'features.csv')
    df = pd.read_csv(path, parse_dates=['datetime_utc'])

    return df

# inference
def generate_predictions(df: pd.DataFrame,
                         model: EnergyLSTM,
                         feature_scaler,
                         target_scaler,
                         start_date: pd.Timestamp,
                         end_date: pd.Timestamp) -> pd.DataFrame:
    """
    Generate LSTM predictions for selected date range

    Args:
        df:           Full feature DataFrame
        model:        Loaded LSTM model
        start_date:   Start of display range
        end_date:     End of display range

    Returns:
        DataFrame with columns [datetime_utc, actual_mw, predicted_mw]
    """
    # get SEQUENCE_LENGTH hours before start_date for 1st prediction
    lookback_start = start_date - pd.Timedelta(hours=SEQUENCE_LENGTH)
    mask = (df['datetime_utc'] >= lookback_start) & (df['datetime_utc'] <= end_date)
    subset = df[mask].reset_index(drop=True)

    if len(subset) <= SEQUENCE_LENGTH:
        st.error('Not enough data for selected time range. Select a wider date range.')
        return pd.DataFrame()
    
    # scale features and target
    features_scaled = feature_scaler.transform(subset[FEATURE_COLS])
    target_scaled = target_scaler.transform(subset[[TARGET_COL]])

    # build sequences and generate predictions
    predictions = []
    model.eval()

    with torch.no_grad():
        for i in range(SEQUENCE_LENGTH, len(subset)):
            seq = features_scaled[i-SEQUENCE_LENGTH:i]
            seq_tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)
            pred_scaled = model(seq_tensor).item()
            pred_mw = target_scaler.inverse_transform([[pred_scaled]])[0][0]
            predictions.append(pred_mw)

    # align predictions with timestamps
    result_df = subset.iloc[SEQUENCE_LENGTH:].copy()
    result_df['predicted_mw'] = predictions
    result_df = result_df.rename(columns = {TARGET_COL: 'actual_mw'})
    result_df = result_df[['datetime_utc', 'actual_mw', 'predicted_mw']]

    # filter to selected date range only
    result_df = result_df[result_df['datetime_utc'] >= start_date].reset_index(drop=True)

    return result_df

# metrics
def compute_metrics(actual: np.ndarray,
                    predicted: np.ndarray) -> dict:
    """
    Compute MAE, RMSE, MAPE
    """
    mae = np.mean(np.abs(actual-predicted))
    rmse = np.sqrt(np.mean((actual-predicted)**2))
    mape = np.mean(np.abs((actual-predicted)/(actual+1e-8)))*100

    return {'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape}

# plots
def plot_forecast(result_df: pd.DataFrame) -> go.Figure:
    """
    Interactive Plotly forecast chart
    Actual vs predicted demand with hover tooltips
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=result_df['datetime_utc'],
                             y=result_df['actual_mw'],
                             name='Actual Demand',
                             line=dict(color='steelblue', width=1.5),
                             hovertemplate='%{x}<br>Actual: %{y:,.0f} MW <extra></extra>'))
    
    fig.add_trace(go.Scatter(x=result_df['datetime_utc'],
                             y=result_df['predicted_mw'],
                             name='LSTM Forecast',
                             line=dict(color='coral', width=1.5),
                             hovertemplate='%{x}<br>Forecast: %{y:,.0f} MW <extra></extra>'))
    
    fig.update_layout(title='MISO Hourly Energy Demand - Actual vs LSTM Forecast',
                      xaxis_title='Date',
                      yaxis_title='Demand (MW)',
                      hovermode='x unified',
                      legend=dict(orientation='h',
                                  yanchor='bottom',
                                  y=1.02,
                                  xanchor='right',
                                  x=1),
                      height=500,
                      template='plotly_white')
    
    return fig

def plot_error_distribution(result_df: pd.DataFrame) -> go.Figure:
    """
    Error distribution histogram for selected period
    """
    errors = result_df['predicted_mw'] - result_df['actual_mw']
    mean_error = errors.mean()

    fig = go.Figure()

    fig.add_trace(go.Histogram(x=errors,
                               nbinsx=50,
                               name='Prediction Error',
                               marker_color='steelblue',
                               opacity=0.75))
    
    fig.add_vline(x=0,
                  line_dash='dash',
                  line_color='red',
                  annotation_text='Zero Error')
    
    fig.add_vline(x=mean_error,
                  line_dash='dash',
                  line_color='orange',
                  annotation_text=f'Mean: {mean_error:,.0f} MW')
    
    fig.update_layout(title='Prediction Error Distribution',
                      xaxis_title='Error (MW)',
                      yaxis_title='Frequency',
                      height=350,
                      template='plotly_white')
    
    return fig

def plot_scatter(result_df: pd.DataFrame) -> go.Figure:
    """
    Scatter plot of predicted vs actual demand
    """
    min_val = min(result_df['actual_mw'].min(),
                  result_df['predicted_mw'].min())
    max_val = max(result_df['actual_mw'].max(),
                  result_df['predicted_mw'].max())
    
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=result_df['actual_mw'],
                             y=result_df['predicted_mw'],
                             mode='markers',
                             marker=dict(color='steelblue',
                                         size=3,
                                         opacity=0.4),
                             name='Predictions',
                             hovertemplate='Actual: %{x:,.0f} MW<br>' \
                                           'Predicted: %{y:,.0f} MW<extra><\extra>'))
    
    # perfect prediction line
    fig.add_trace(go.Scatter(x=[min_val, max_val],
                             y=[min_val, max_val],
                             mode='lines',
                             line=dict(color='red',
                                       dash='dash',
                                       width=1.5),
                             name='Perfect Forecast'))
    
    fig.update_layout(title='Predicted vs Actual Demand',
                      xaxis_title='Actual (MW)',
                      yaxis_title='Predicted (MW)',
                      height=350,
                      template='plotly_white')
    
    return fig

# sidebar
def render_sidebar(df: pd.DataFrame) -> tuple:
    """
    Render sidebar controls and returns selected date range
    Contains all ux controls
    """
    st.sidebar.header('Forecast Controls')
    st.sidebar.markdown('---')

    # test set date range - only show these test set dates and they are unseen by the model
    test_start = pd.Timestamp('2023-07-21')
    test_end = pd.Timestamp('2023-12-30')

    st.sidebar.markdown('**Select Date Range**')
    st.sidebar.caption('Dates are from the held-out test set -'
                       'data the model never saw during training.')
    
    start_date = st.sidebar.date_input('Start Date',
                                       value=test_start.date(),
                                       min_value=test_start.date(),
                                       max_value=test_end.date())
    
    end_date = st.sidebar.date_input('End Date',
                                     value=(test_start+pd.Timedelta(days=14)).date(),
                                     min_value=test_start.date(),
                                     max_value=test_end.date())
    
    st.sidebar.markdown('**Overall Test Set Performance**')
    st.sidebar.table({'Model': ['Persistence', 'Random Forest', 'LSTM'],
                      'MAPE': ['4.30%', '2.26%', '1.82%']})
    
    return pd.Timestamp(start_date), pd.Timestamp(end_date)

# main app
def main():
    """
    Main app
    Streamlit reruns this function on every user interaction; caching prevents expensive operations from re-running each time 
    """
    # header
    st.title('MISO Energy Demand Forecasting')
    st.markdown("**PyTorch LSTM Model forecasting hourly electricity demand "
                "for the Midcontinent Independent System Operator (MISO) region.** "
                "Trained on 3 years of EIA hourly demand data paired with "
                "Chicago O'Hare weather observations.")
    
    # headline metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric('MAPE', '1.82%', 
                delta='-2.48% vs Baseline',
                delta_color = 'inverse')
    col2.metric('MAE',
                '1.351 MW')
    col3.metric('RMSE',
                '1,800 MW')
    col4.metric('Training Records',
                '26,068')
    
    st.markdown('---')

    # load resources
    with st.spinner('Loading model and data...'):
        model = load_model()
        feature_scaler, target_scaler = load_scalers()
        df = load_data()

    # sidebar
    start_date, end_date = render_sidebar(df)

    # validate date range
    if start_date >= end_date:
        st.error('Start date must be before end date.')
        st.stop()

    # generate forecast
    with st.spinner('Generating forecast...'):
        result_df = generate_predictions(df,
                                         model,
                                         feature_scaler,
                                         target_scaler,
                                         start_date,
                                         end_date)

    # check if result df is empty and if True, stop
    if result_df.empty:
        st.stop()

    # selected period metrics
    metrics = compute_metrics(result_df['actual_mw'].values,
                              result_df['predicted_mw'].values)
    
    st.subheader(f"Selected Period: {start_date.strftime('%b %d')} - {end_date.strftime('%b %d, %Y')}")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric('MAPE', f"{metrics['MAPE']:.2f}%")
    m2.metric('MAE', f"{metrics['MAE']:,.0f} MW")
    m3.metric('RMSE', f"{metrics['RMSE']:,.0f} MW")
    m4.metric('Hours Forecast', f"{len(result_df):,}")

    # forecast chart
    st.plotly_chart(plot_forecast(result_df), use_container_width=True)

    # bottom charts
    col_left, col_right = st.columns(2)

    with col_left:
        st.plotly_chart(plot_error_distribution(result_df), use_container_width=True)

    with col_left:
        st.plotly_chart(plot_scatter(result_df), use_container_width=True)

    # footer
    st.markdown('---')
    st.caption('Data source: U.S. Energy Information Administration (EIA)'
               'Weather: Iowa Environmental Mesonet (IEM)'
               'Model: PyTorch LSTM'
               'Built by Jeremy Reinert')
    
if __name__ == '__main__':
    main()