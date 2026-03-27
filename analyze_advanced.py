"""
Advanced time series analysis with forecasting using statistical methods.
Uses SARIMA, exponential smoothing, and trend analysis.
"""

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Input CSV from previous script
input_csv = "/Users/sidchaudhary/Downloads/pixel_level_ET_timeseries.csv"

# Output CSVs
output_trends_csv = "/Users/sidchaudhary/Documents/GitHub/ET-TimeSeries Foundation Model/field_advanced_analysis.csv"
output_forecasts_csv = "Users/sidchaudhary/Documents/GitHub/ET-TimeSeries Foundation Model/field_forecasts.csv"

print("="*60)
print("Advanced Time Series Analysis with Forecasting")
print("="*60)

# Try to import statsmodels for advanced forecasting
try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from statsmodels.tsa.seasonal import seasonal_decompose
    STATSMODELS_AVAILABLE = True
    print("\nUsing statsmodels for advanced forecasting")
except ImportError:
    STATSMODELS_AVAILABLE = False
    print("\nstatsmodels not available - using basic forecasting")
    print("Install with: pip install statsmodels")

print("\nLoading data...")
df = pd.read_csv(input_csv)

print(f"Total records: {len(df)}")
print(f"Unique fields: {df['field_id'].nunique()}")
print(f"Date range: {df['year'].min()}-{df['year'].max()}")

# Aggregate to field level
print("\nAggregating to field level...")
field_agg = df.groupby(['field_id', 'year', 'month']).agg({
    'ET': 'mean',
    'relative_ET': 'mean'
}).reset_index()

field_agg['time_period'] = pd.to_datetime(
    field_agg['year'].astype(str) + '-' + field_agg['month'].astype(str).str.zfill(2) + '-01'
)
field_agg = field_agg.sort_values(['field_id', 'time_period'])

print(f"Aggregated records: {len(field_agg)}")

# Analyze each field
results = []
forecasts_list = []

print("\nAnalyzing fields with forecasting...")

for field_id in tqdm(field_agg['field_id'].unique(), desc="Processing fields"):
    field_data = field_agg[field_agg['field_id'] == field_id].copy()
    field_data = field_data.sort_values('time_period')
    
    et_values = field_data['ET'].values
    time_periods = field_data['time_period'].values
    
    if len(et_values) < 8:
        continue
    
    # Basic statistics
    mean_et = np.mean(et_values)
    std_et = np.std(et_values)
    
    # Calculate trend
    time_index = np.arange(len(et_values))
    trend_coef = np.polyfit(time_index, et_values, 1)[0]
    trend_pct = (trend_coef / mean_et) * 100 if mean_et > 0 else 0
    
    # Calculate persistence (autocorrelation)
    if len(et_values) > 2 and std_et > 0:
        et_norm = (et_values - mean_et) / std_et
        persistence = np.corrcoef(et_norm[:-1], et_norm[1:])[0, 1]
    else:
        persistence = 0
    
    # Forecasting
    forecast_values = None
    forecast_mean = None
    forecast_trend = None
    forecast_uncertainty = None
    anomaly_score = 0
    seasonality_strength = 0
    
    # Try exponential smoothing forecast
    if STATSMODELS_AVAILABLE and len(et_values) >= 12:
        try:
            # Fit exponential smoothing model
            model = ExponentialSmoothing(
                et_values,
                seasonal_periods=4,  # 4 months per year in our data
                trend='add',
                seasonal='add',
                initialization_method='estimated'
            )
            fitted_model = model.fit()
            
            # Generate 12-step forecast
            forecast_values = fitted_model.forecast(steps=12)
            forecast_mean = np.mean(forecast_values)
            forecast_uncertainty = np.std(forecast_values)
            
            # Calculate forecast trend
            forecast_time_index = np.arange(len(forecast_values))
            forecast_trend = np.polyfit(forecast_time_index, forecast_values, 1)[0]
            
            # Detect seasonality
            if len(et_values) >= 16:
                try:
                    decomposition = seasonal_decompose(et_values, model='additive', period=4, extrapolate_trend='freq')
                    seasonality_strength = np.std(decomposition.seasonal) / std_et if std_et > 0 else 0
                except:
                    seasonality_strength = 0
            
            # Store forecasts
            for i, fval in enumerate(forecast_values):
                forecasts_list.append({
                    'field_id': field_id,
                    'forecast_step': i + 1,
                    'forecast_ET': float(fval)
                })
        
        except Exception as e:
            # Fallback to simple linear extrapolation
            forecast_time_index = np.arange(len(et_values), len(et_values) + 12)
            forecast_values = trend_coef * forecast_time_index + (mean_et - trend_coef * np.mean(time_index))
            forecast_mean = np.mean(forecast_values)
            forecast_trend = trend_coef
            forecast_uncertainty = std_et
            
            for i, fval in enumerate(forecast_values):
                forecasts_list.append({
                    'field_id': field_id,
                    'forecast_step': i + 1,
                    'forecast_ET': float(fval)
                })
    
    else:
        # Simple linear extrapolation
        forecast_time_index = np.arange(len(et_values), len(et_values) + 12)
        forecast_values = trend_coef * forecast_time_index + (mean_et - trend_coef * np.mean(time_index))
        forecast_mean = np.mean(forecast_values)
        forecast_trend = trend_coef
        forecast_uncertainty = std_et
        
        for i, fval in enumerate(forecast_values):
            forecasts_list.append({
                'field_id': field_id,
                'forecast_step': i + 1,
                'forecast_ET': float(fval)
            })
    
    # Anomaly detection
    recent_values = et_values[-4:]
    anomaly_score = np.abs(np.mean(recent_values) - mean_et) / std_et if std_et > 0 else 0
    
    # Trend classification
    if abs(trend_pct) < 0.5:
        trend_direction = "stable"
    elif trend_pct > 0:
        trend_direction = "increasing"
    else:
        trend_direction = "decreasing"
    
    # Persistence classification
    if persistence > 0.7:
        persistence_class = "high"
    elif persistence > 0.4:
        persistence_class = "moderate"
    else:
        persistence_class = "low"
    
    # Anomaly classification
    if anomaly_score > 2:
        anomaly_status = "high_anomaly"
    elif anomaly_score > 1:
        anomaly_status = "moderate_anomaly"
    else:
        anomaly_status = "normal"
    
    results.append({
        'field_id': field_id,
        'n_observations': len(et_values),
        'mean_ET': mean_et,
        'std_ET': std_et,
        'cv_percent': (std_et / mean_et * 100) if mean_et > 0 else 0,
        'trend_coefficient': trend_coef,
        'trend_percent_per_period': trend_pct,
        'trend_direction': trend_direction,
        'persistence_lag1': persistence,
        'persistence_class': persistence_class,
        'seasonality_strength': seasonality_strength,
        'anomaly_score': anomaly_score,
        'anomaly_status': anomaly_status,
        'forecast_mean_ET': forecast_mean,
        'forecast_trend': forecast_trend,
        'forecast_uncertainty': forecast_uncertainty,
        'min_ET': np.min(et_values),
        'max_ET': np.max(et_values),
        'range_ET': np.max(et_values) - np.min(et_values)
    })

# Create results DataFrame
results_df = pd.DataFrame(results)

print(f"\nAnalysis complete!")
print(f"Fields analyzed: {len(results_df)}")

# Save results
results_df.to_csv(output_trends_csv, index=False)
print(f"\nTrends and persistence saved to: {output_trends_csv}")

if len(forecasts_list) > 0:
    forecasts_df = pd.DataFrame(forecasts_list)
    forecasts_df.to_csv(output_forecasts_csv, index=False)
    print(f"Forecasts saved to: {output_forecasts_csv}")
    print(f"Total forecast records: {len(forecasts_df)}")

# Display summary
print("\n" + "="*60)
print("SUMMARY STATISTICS")
print("="*60)
print(f"\nTrend distribution:")
print(results_df['trend_direction'].value_counts())
print(f"\nPersistence distribution:")
print(results_df['persistence_class'].value_counts())
print(f"\nAnomaly distribution:")
print(results_df['anomaly_status'].value_counts())

print(f"\nMean ET: {results_df['mean_ET'].mean():.2f} ± {results_df['mean_ET'].std():.2f}")
print(f"Mean persistence: {results_df['persistence_lag1'].mean():.3f}")
print(f"Mean trend: {results_df['trend_percent_per_period'].mean():.3f}% per period")
print(f"Mean seasonality strength: {results_df['seasonality_strength'].mean():.3f}")

if results_df['forecast_mean_ET'].notna().any():
    print(f"\nMean forecast ET: {results_df['forecast_mean_ET'].mean():.2f}")
    print(f"Mean forecast uncertainty: {results_df['forecast_uncertainty'].mean():.2f}")
    print(f"Fields with forecasts: {results_df['forecast_mean_ET'].notna().sum()}")
    
    # Compare historical vs forecast
    historical_mean = results_df['mean_ET'].mean()
    forecast_mean_avg = results_df['forecast_mean_ET'].mean()
    change_pct = ((forecast_mean_avg - historical_mean) / historical_mean) * 100
    print(f"\nHistorical mean ET: {historical_mean:.2f}")
    print(f"Forecast mean ET: {forecast_mean_avg:.2f}")
    print(f"Expected change: {change_pct:+.2f}%")

print("\n" + "="*60)
print("\nTop 10 fields with highest forecast ET:")
top_forecast = results_df.nlargest(10, 'forecast_mean_ET')[['field_id', 'mean_ET', 'forecast_mean_ET', 'trend_direction', 'persistence_class']]
print(top_forecast.to_string(index=False))

print("\n" + "="*60)
print("\nFields with anomalies:")
anomalies = results_df[results_df['anomaly_status'] != 'normal'][['field_id', 'mean_ET', 'anomaly_score', 'anomaly_status']]
if len(anomalies) > 0:
    print(anomalies.to_string(index=False))
else:
    print("No anomalies detected")

print("\n" + "="*60)
