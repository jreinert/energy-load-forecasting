# -*- coding: utf-8 -*-
"""
evaluate.py
Evaluates the trained LSTM model on the test set.
Computes MAE, RMSE, and MAPE metrics.
Compares against persistence baseline and Random Forest benchmark.
Generates evaluation visualizations.
"""

# import libs
import os
import sys
import numpy as np
import pandas as pd
import torch
import joblib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

sys.path.insert(0, os.path.dirname(__file__))

from model import EnergyLSTM
from dataset import (split_data, 
                     fit_scalers, 
                     build_dataloaders,
                     apply_scalers, 
                     EnergyDataset,
                     FEATURE_COLS, 
                     TARGET_COL, 
                     SEQUENCE_LENGTH, 
                     BATCH_SIZE)

# function declarations
# metrics
def mean_absolute_percentage_error(actual: np.ndarray,
                                   predicted: np.ndarray) -> float:
    """
    Computes MAPE - expressed as a %
    """
    # epsilon added to avoid division by 0 in case any actual values are 0
    epsilon = 1e-8
    return np.mean(np.abs((actual-predicted) / (actual+epsilon))) * 100

def compute_metrics(actual: np.ndarray,
                    predicted: np.ndarray,
                    model_name: str) -> dict:
    """
    Computes and prints MAE, RMSE, and MAPE for a model

    Args:
        actual:     True demand values in MW
        predicted:  Predicted demand values in MW
        model_name: Label for printing

    Returns:
        Dictionary of metric name → value
    """
    # calculate values
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mape = mean_absolute_percentage_error(actual, predicted)

    # print metrics
    print(f'\n{model_name}')
    print(f'  MAE:  {mae:,.1f} MW')
    print(f'  RMSE: {rmse:,.1f} MW')
    print(f'  MAPE: {mape:.2f}%')

    # return dict with metrics
    return {'model': model_name, 'MAE': mae, 'RMSE': rmse, 'MAPE': mape}

# lstm predictions
def get_lstm_predictions(model: torch.nn.Module,
                         loader,
                         target_scaler,
                         device: torch.device) -> tuple:
    """
    Generates predictions from the trained LSTM on the test set
    Steps:
    1. Run forward pass on all test batches
    2. Collect scaled predictions and scaled actuals
    3. Inverse transform both back to real MW values - need both actuals and targets on the same scale for metric computation

    Returns:
        Tuple of (actual_mw, predicted_mw) as numpy arrays
    """
    model.eval()
    all_predictions = []
    all_actuals = []

    with torch.no_grad():
        for sequences, targets in loader:
            sequences = sequences.to(device)
            predictions = model(sequences)

            # move to cpu and convert to numpy
            all_predictions.extend(predictions.squeeze().cpu().numpy())
            all_actuals.extend(targets.numpy())

    # convert to numpy arrays and reshape for inverse transform
    all_predictions = np.array(all_predictions).reshape(-1,1)
    all_actuals = np.array(all_actuals).reshape(-1,1)

    # inverse transform from scaled units back to mw
    predicted_mw = target_scaler.inverse_transform(all_predictions).flatten()
    actual_mw = target_scaler.inverse_transform(all_actuals).flatten()

    # return actual_mw & predicted_mw
    return actual_mw, predicted_mw

# baseline - persistence model
def get_persistence_predictions(test_df: pd.DataFrame) -> tuple:
    """
    Persistence baseline - predicts next hour == current hour

    Persistence forecast for hour t is the actual value at t-1. Using lag_24 rather than lag_1 because its a farier baseline for a 24 hr ahead forecast
    """
    actual = test_df[TARGET_COL].values[SEQUENCE_LENGTH:]
    predicted = test_df['lag_24'].values[SEQUENCE_LENGTH:]

    # return tuple values
    return actual, predicted

# benchmark - random forest
def get_rf_predictions(train_df: pd.DataFrame,
                       test_df: pd.DataFrame,
                       feature_scaler,
                       target_scaler) -> tuple:
    """
    Trains a Random Forest on the same features and generates predictions
    """
    print('\nTraining Random Forest benchmark...')
    
    # scale the data
    train_scaled = apply_scalers(train_df, feature_scaler, target_scaler)
    test_scaled = apply_scalers(test_df, feature_scaler, target_scaler)

    X_train = train_scaled[FEATURE_COLS].values
    y_train = train_scaled[TARGET_COL].values
    X_test = test_scaled[FEATURE_COLS].values
    y_test = test_scaled[TARGET_COL].values

    # train RF - n_estimators = 100, n_jobs = -1 (uses all cpu cores for faster training)
    rf = RandomForestRegressor(n_estimators=100,
                               random_state=42,
                               n_jobs=-1)
    rf.fit(X_train, y_train)

    # generate predictions and inverse transform
    rf_pred_scaled = rf.predict(X_test).reshape(-1, 1)
    rf_actual_scaled = y_test.reshape(-1, 1)

    predicted_mw = target_scaler.inverse_transform(rf_pred_scaled).flatten()
    actual_mw = target_scaler.inverse_transform(rf_actual_scaled).flatten()

    print('Random Forest training complete')

    # return tuple vals
    return actual_mw, predicted_mw

# visualizations
def plot_predictions(actual: np.ndarray,
                     predicted: np.ndarray,
                     test_df: pd.DataFrame,
                     model_name: str = 'LSTM'):
    """
    3 panel eval plot:
    1. Pred vs Actual time series (2wk window)
    2. Scatter Plot - pred vs actual
    3. Error dist histogram
    """
    # use timestamps from test_df aligned to predictions
    timestamps = pd.to_datetime(test_df['datetime_utc'].values[SEQUENCE_LENGTH:])

    # calculate errors
    errors = predicted - actual

    fig = plt.figure(figsize=(16,12))
    gs = gridspec.GridSpec(2,2,figure=fig)

    # plot 1: full test period overview
    ax1 = fig.add_subplot(gs[0,:])
    ax1.plot(timestamps,
             actual,
             label='Actual',
             color='steelblue',
             alpha=0.8,
             linewidth=0.8)
    ax1.plot(timestamps,
             predicted,
             label=f'{model_name} Forecast',
             color='coral',
             alpha=0.8,
             linewidth=0.8)
    ax1.set_title(f'{model_name} - Full Test Period (Jul-Dec 2023)',
                  fontsize=13)
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Demand (MW)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # plot 2: 2-week zoom
    ax2 = fig.add_subplot(gs[1,0])
    zoom = 336 # 2 weeks = 14 days * 24 hrs
    ax2.plot(timestamps[:zoom],
             actual[:zoom],
             label='Actual',
             color='steelblue',
             linewidth=1.2)
    ax2.plot(timestamps[:zoom],
             predicted[:zoom],
             label=f'{model_name} Forecast',
             color='coral',
             linewidth=1.2)
    ax2.set_title('2-Week Zoom (First 2 Weeks of Test Period)',
                  fontsize=11)
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Demand (MW)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.setp(ax2.xaxis.get_majorticklabels(),rotation=30)

    # plot 3: error distribution
    ax3 = fig.add_subplot(gs[1,1])
    ax3.hist(errors,
             bins=60,
             color='steelblue',
             alpha=0.7,
             edgecolor='white')
    ax3.axvline(x=0,
                color='red',
                linestyle='--',
                linewidth=1.5,
                label=f'Mean error: {np.mean(errors):,.0f} MW')
    ax3.set_title('Prediction Error Distribution',
                  fontsize=11)
    ax3.set_xlabel('Error (MW)')
    ax3.set_ylabel('Frequency')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.suptitle(f'MISO Energy Demand Forecasting - {model_name} Evaluation',
                  fontsize=14,
                  fontweight='bold',
                  y=1.01)
    plt.tight_layout()

    # save plot to png
    output_path = os.path.join('models', f'evaluation_{model_name.lower()}.png')
    plt.savefig(output_path,
                dpi=150,
                bbox_inches='tight')
    
    # show plot
    plt.show()
    print(f'Evaluation plot saved to {output_path}')

def plot_model_comparison(results: list):
    """
    Bar chart comparing MAE, RMSE, and MAPE across all 3 models
    """
    # create df from the results list param
    df = pd.DataFrame(results)

    fig, axes = plt.subplots(1, 3, figsize=(14,5))
    colors = ['#d9534f', '#f0ad4e', '#5cb85c']
    metrics = ['MAE', 'RMSE', 'MAPE']
    labels = ['MAE (MW)', 'RMSE (MW)', 'MAPE (%)']

    for ax, metric, label, color in zip(axes, metrics, labels, colors):
        bars = ax.bar(df['model'],
                      df[metric],
                      color=color,
                      alpha=0.8,
                      edgecolor='white',
                      linewidth=1.2)
        ax.set_title(metric,
                     fontsize=12,
                     fontweight='bold')
        ax.set_ylabel(label)
        ax.grid(True,
                alpha=0.3,
                axis='y')
        
        # add value labels on bars
        for bar, val in zip(bars, df[metric]):
            if metric == 'MAPE':
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.1,
                        f'{val:.2f}',
                        ha='center',
                        va='bottom',
                        fontsize=10)
            else:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 10,
                        f'{val:.2f}',
                        ha='center',
                        va='bottom',
                        fontsize=10)
    
    plt.suptitle('Model Comparison - MISO Energy Demand Forecasting',
                 fontsize=13,
                 fontweight='bold')
    plt.tight_layout()

    # save plot as png
    output_path = os.path.join('models', 'model_comparison.png')
    plt.savefig(output_path, 
                dpi= 150,
                bbox_inches='tight')
    
    # show plot
    plt.show()
    print(f'Model comparison plot saved to {output_path}')

# main evaluation pipeline
def evaluate():
    """
    Full evaluation pipeline - loads trained model, generates predictions, computes metrics, compares against baselines, and saves visualizations
    """
    # set device
    device = torch.device('cpu')

    # load data
    print('Loading feature matrix...')
    df = pd.read_csv(os.path.join('data', 'processed', 'features.csv'), parse_dates=['datetime_utc'])

    # create train, val, test dfs
    train_df, val_df, test_df = split_data(df)
    
    # create feature and target scalerse
    feature_scaler, target_scaler = fit_scalers(train_df)

    # create data loaders
    train_loader, val_loader, test_loader = build_dataloaders(train_df,
                                                              val_df,
                                                              test_df,
                                                              feature_scaler,
                                                              target_scaler)
    
    # load trained LSTM
    print('\nLoading trained LSTM model...')
    model = EnergyLSTM(input_size=13,
                       hidden_size=128,
                       num_layers=2,
                       dropout=0.2,
                       output_size=1)
    
    model.load_state_dict(
        torch.load(os.path.join('models', 'best_model.pt'),
                   map_location=device,
                   weights_only=True)
    )
    model.to(device)
    print('Model loaded successfully')

    # get LSTM predictions
    print('\nGenerating LSTM predictions on test set...')
    lstm_actual, lstm_predicted = get_lstm_predictions(model,
                                                       test_loader,
                                                       target_scaler,
                                                       device)
    
    # get baseline predictions
    persistence_actual, persistence_predicted = get_persistence_predictions(test_df)

    # align lengths - persistence uses lag_24 so lengths may differ slightly
    min_len = min(len(lstm_actual), len(persistence_actual))
    lstm_actual = lstm_actual[:min_len]
    lstm_predicted = lstm_predicted[:min_len]
    persistence_actual = persistence_actual[:min_len]
    persistence_predicted = persistence_predicted[:min_len]

    # get random forest predictions
    rf_actual, rf_predicted = get_rf_predictions(train_df,
                                                 test_df,
                                                 feature_scaler,
                                                 target_scaler)
    # align lengths
    rf_actual = rf_actual[:min_len]
    rf_predicted = rf_predicted[:min_len]

    # compute metrics
    print('\n' + '=' * 50)
    print('Test Set Evaluation Results')
    print('=' * 50)

    results = []
    results.append(compute_metrics(persistence_actual,
                                   persistence_predicted,
                                   'Persistence Baseline'))
    
    results.append(compute_metrics(rf_actual,
                                   rf_predicted,
                                   'Random Forest'))
    
    results.append(compute_metrics(lstm_actual,
                                   lstm_predicted,
                                   'LSTM'))
    
    # generate visualizations
    print('\nGenerating Evaluation plots...')
    plot_predictions(lstm_actual,
                     lstm_predicted,
                     test_df,
                     'LSTM')
    plot_model_comparison(results)

    # summary table
    print('\n' + '=' * 50)
    print('Summary')
    print('=' * 50)
    
    # create df from results list
    results_df = pd.DataFrame(results)
    results_df['MAE'] = results_df['MAE'].map('{:,.1f} MW'.format)
    results_df['RMSE'] = results_df['RMSE'].map('{:,.1f} MW'.format)
    results_df['MAPE'] = results_df['MAPE'].map('{:,.2f}%'.format)
    print(results_df.to_string(index=False))

    # retun results
    return results

if __name__ == '__main__':
    evaluate()
