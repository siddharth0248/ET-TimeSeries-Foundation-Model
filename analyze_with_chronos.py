"""
Advanced time series analysis using Amazon's Chronos foundation model.
Chronos is compatible with Python 3.9+ and provides similar capabilities to TimesFM.
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
output_trends_csv = "/Users/sidchaudhary/Downloads/field_chronos_analysis.csv"
output_forecasts_csv = "/Users/sidchaudhary/Downloads/field_chronos_forecasts.csv"

print("="*60)
print("Chronos-based Time Series Analysis")
print("="*60)

# Try to import Chronos
try:
    import torch
    from chronos import ChronosPipeline
    
    print("\nInitializing Chronos model...")
    print("Loading chronos-t5-small model (this may take a moment)...")
    
    # Initialize Chronos pipeline
    # Available models: chronos-t5-tiny, chronos-t5-mini, chronos-t5-small, chronos-t5-base, chronos-t5-large
    pipeline = ChronosPipeline.from_pretrained(
        "amazon/chronos-t5-small",
        device_map="cpu",  # Use CPU (change to "cuda" if GPU available)
        torch_dtype=torch.float32,
    )
    
    CHRONOS_AVAILABLE = True
    print("Chronos model loaded successfully!")
    
except ImportError as e:
    CHRONOS_AVAILABLE = False
    print("\n" + "!"*60)
    print("ERROR: Chronos not installed!")
    print("!"*60)
    print("\nTo install Chronos, run:")
    print("  pip install git+https://github.com/amazon-science/chronos-forecasting.git")
    print("\nOr install dependencies:")
    print("  pip install torch transformers")
    print("\nFalling back to statistical methods...")
    print("="*60 + "\n")

except Exception as e:
    CHRONOS_AVAILABLE = False
    print(f"\nWarning: Could not load Chronos model: {e}")
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
    
    # Chronos forecasting (if available)
    forecast_values = None
    forecast_mean = None
    forecast_trend = None
    anomaly_score = 0
    forecast_uncertainty = None
    
    if CHRONOS_AVAILABLE and len(et_values) >= 12:
        try:
            # Prepare data for Chronos
            context = torch.tensor(et_values, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
            
            # Generate forecast (12 steps ahead, 100 samples for uncertainty)
            forecast_result = pipeline.predict(
                context,
                prediction_length=12,
                num_samples=100,
            )
            
            # Extract forecast statistics
            forecast_samples = forecast_result[0].numpy()  # Shape: (num_samples, prediction_length)
            forecast_values = np.median(forecast_samples, axis=0)  # Median forecast
            forecast_mean = np.mean(forecast_values)
            forecast_uncertainty = np.std(forecast_samples, axis=0).mean()  # Average uncertainty
            
            # Calculate forecast trend
            forecast_time_index = np.arange(len(forecast_values))
            forecast_trend = np.polyfit(forecast_time_index, forecast_values, 1)[0]
            
            # Anomaly detection: compare recent values to historical mean
            recent_values = et_values[-4:]  # Last 4 observations
            anomaly_score = np.abs(np.mean(recent_values) - mean_et) / std_et if std_et > 0 else 0
            
            # Store forecasts with confidence intervals
            forecast_lower = np.percentile(forecast_samples, 10, axis=0)
            forecast_upper = np.percentile(forecast_samples, 90, axis=0)
            
            for i in range(len(forecast_values)):
                forecasts_list.append({
                    'field_id': field_id,
                    'forecast_step': i + 1,
                    'forecast_ET': float(forecast_values[i]),
                    'forecast_lower_10': float(forecast_lower[i]),
                    'forecast_upper_90': float(forecast_upper[i]),
                })
        
        except Exception as e:
            print(f"\nWarning: Chronos forecast failed for field {field_id}: {e}")
            forecast_mean = None
            forecast_trend = None
            forecast_uncertainty = None
    
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

if 'anomaly_status' in results_df.columns:
    print(f"\nAnomaly distribution:")
    print(results_df['anomaly_status'].value_counts())

print(f"\nMean ET: {results_df['mean_ET'].mean():.2f} ± {results_df['mean_ET'].std():.2f}")
print(f"Mean persistence: {results_df['persistence_lag1'].mean():.3f}")
print(f"Mean trend: {results_df['trend_percent_per_period'].mean():.3f}% per period")

if CHRONOS_AVAILABLE and results_df['forecast_mean_ET'].notna().any():
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
print("\nTop 5 fields with highest forecast ET:")
if CHRONOS_AVAILABLE and results_df['forecast_mean_ET'].notna().any():
    top_forecast = results_df.nlargest(5, 'forecast_mean_ET')[['field_id', 'mean_ET', 'forecast_mean_ET', 'trend_direction']]
    print(top_forecast.to_string(index=False))

print("\n" + "="*60)
