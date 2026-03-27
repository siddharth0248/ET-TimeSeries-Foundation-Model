"""
Advanced time series analysis using Google's TimesFM foundation model.
This script uses TimesFM for forecasting and anomaly detection.
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
output_trends_csv = "/Users/sidchaudhary/Documents/GitHub/ET-TimeSeries Foundation Model/field_timesfm_analysis.csv"
output_forecasts_csv = "/Users/sidchaudhary/Documents/GitHub/ET-TimeSeries Foundation Model/field_timesfm_forecasts.csv"

print("="*60)
print("TimesFM-based Time Series Analysis")
print("="*60)

# Try to import and initialize TimesFM
try:
    import timesfm
    
    print("\nInitializing TimesFM model...")
    # Initialize TimesFM model
    # Available models: timesfm-1.0-200m (default)
    tfm = timesfm.TimesFm(
        context_len=512,  # Context length for the model
        horizon_len=12,   # Forecast horizon (12 periods ahead)
        input_patch_len=32,
        output_patch_len=128,
        num_layers=20,
        model_dims=1280,
    )
    
    # Load checkpoint
    tfm.load_from_checkpoint(repo_id="google/timesfm-1.0-200m")
    
    TIMESFM_AVAILABLE = True
    print("TimesFM model loaded successfully!")
    
except ImportError:
    TIMESFM_AVAILABLE = False
    print("\n" + "!"*60)
    print("ERROR: TimesFM not installed!")
    print("!"*60)
    print("\nTo install TimesFM, run:")
    print("  pip install timesfm")
    print("\nOr with specific dependencies:")
    print("  pip install git+https://github.com/google-research/timesfm.git")
    print("\nFalling back to statistical methods...")
    print("="*60 + "\n")

except Exception as e:
    TIMESFM_AVAILABLE = False
    print(f"\nWarning: Could not load TimesFM model: {e}")
    print("Falling back to statistical methods...\n")

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

print("\nAnalyzing fields...")

for field_id in tqdm(field_agg['field_id'].unique(), desc="Processing fields"):
    field_data = field_agg[field_agg['field_id'] == field_id].copy()
    field_data = field_data.sort_values('time_period')
    
    et_values = field_data['ET'].values
    time_periods = field_data['time_period'].values
    
    if len(et_values) < 5:
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
    
    # TimesFM forecasting (if available)
    forecast_values = None
    forecast_mean = None
    forecast_trend = None
    anomaly_score = 0
    
    if TIMESFM_AVAILABLE and len(et_values) >= 12:
        try:
            # Prepare data for TimesFM (needs to be 2D: [batch, time])
            input_data = et_values.reshape(1, -1).astype(np.float32)
            
            # Generate forecast
            forecast_result = tfm.forecast(
                inputs=input_data,
                freq=[0],  # Frequency indicator (0 for unknown/mixed)
            )
            
            # Extract forecast
            forecast_values = forecast_result[0].flatten()
            forecast_mean = np.mean(forecast_values)
            
            # Calculate forecast trend
            forecast_time_index = np.arange(len(forecast_values))
            forecast_trend = np.polyfit(forecast_time_index, forecast_values, 1)[0]
            
            # Anomaly detection: compare recent values to historical mean
            recent_values = et_values[-4:]  # Last 4 observations
            anomaly_score = np.abs(np.mean(recent_values) - mean_et) / std_et if std_et > 0 else 0
            
            # Store forecasts
            for i, fval in enumerate(forecast_values):
                forecasts_list.append({
                    'field_id': field_id,
                    'forecast_step': i + 1,
                    'forecast_ET': float(fval)
                })
        
        except Exception as e:
            print(f"Warning: TimesFM forecast failed for field {field_id}: {e}")
            forecast_mean = None
            forecast_trend = None
    
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
        'anomaly_score': anomaly_score,
        'anomaly_status': anomaly_status,
        'forecast_mean_ET': forecast_mean,
        'forecast_trend': forecast_trend,
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

# Display summary
print("\n" + "="*60)
print("SUMMARY STATISTICS")
print("="*60)
print(f"\nTrend distribution:")
print(results_df['trend_direction'].value_counts())
print(f"\nPersistence distribution:")
print(results_df['persistence_class'].value_counts())

if 'anomaly_status' in results_df.columns:
    print(f"\nAnomaly distribution:")
    print(results_df['anomaly_status'].value_counts())

print(f"\nMean ET: {results_df['mean_ET'].mean():.2f} ± {results_df['mean_ET'].std():.2f}")
print(f"Mean persistence: {results_df['persistence_lag1'].mean():.3f}")
print(f"Mean trend: {results_df['trend_percent_per_period'].mean():.3f}% per period")

if TIMESFM_AVAILABLE and results_df['forecast_mean_ET'].notna().any():
    print(f"\nMean forecast ET: {results_df['forecast_mean_ET'].mean():.2f}")
    print(f"Fields with forecasts: {results_df['forecast_mean_ET'].notna().sum()}")

print("\n" + "="*60)
