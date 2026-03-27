import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Try to import timesfm
try:
    import timesfm
    TIMESFM_AVAILABLE = True
except ImportError:
    TIMESFM_AVAILABLE = False
    print("TimesFM not available. Install with: pip install timesfm")

# Input CSV from previous script
input_csv = "/Users/sidchaudhary/Downloads/pixel_level_ET_timeseries.csv"

# Output CSV
output_csv = "/Users/sidchaudhary/Documents/GitHub/ET-TimeSeries Foundation Model/field_persistence_trends.csv"

print("Loading data...")
df = pd.read_csv(input_csv)

print(f"Total records: {len(df)}")
print(f"Unique fields: {df['field_id'].nunique()}")
print(f"Date range: {df['year'].min()}-{df['year'].max()}")
print(f"Months: {sorted(df['month'].unique())}")

# Aggregate to field level (mean ET per field per time period)
print("\nAggregating to field level...")
field_agg = df.groupby(['field_id', 'year', 'month']).agg({
    'ET': 'mean',
    'relative_ET': 'mean'
}).reset_index()

# Create time period column
field_agg['time_period'] = field_agg['year'].astype(str) + '-' + field_agg['month'].astype(str).str.zfill(2)
field_agg = field_agg.sort_values(['field_id', 'year', 'month'])

print(f"Aggregated records: {len(field_agg)}")

# Calculate persistence and trends for each field
results = []

print("\nAnalyzing trends and persistence for each field...")

for field_id in tqdm(field_agg['field_id'].unique(), desc="Processing fields"):
    field_data = field_agg[field_agg['field_id'] == field_id].copy()
    
    # Sort by time
    field_data = field_data.sort_values(['year', 'month'])
    
    et_values = field_data['ET'].values
    years = field_data['year'].values
    
    # Skip if insufficient data
    if len(et_values) < 3:
        continue
    
    # Calculate basic statistics
    mean_et = np.mean(et_values)
    std_et = np.std(et_values)
    cv = (std_et / mean_et) * 100 if mean_et > 0 else 0  # Coefficient of variation
    
    # Calculate linear trend
    time_index = np.arange(len(et_values))
    if len(time_index) > 1:
        trend_coef = np.polyfit(time_index, et_values, 1)[0]
        # Normalize trend by mean
        trend_pct_per_period = (trend_coef / mean_et) * 100 if mean_et > 0 else 0
    else:
        trend_coef = 0
        trend_pct_per_period = 0
    
    # Calculate persistence (autocorrelation at lag 1)
    if len(et_values) > 1:
        # Lag-1 autocorrelation
        et_normalized = (et_values - mean_et) / std_et if std_et > 0 else et_values
        persistence = np.corrcoef(et_normalized[:-1], et_normalized[1:])[0, 1] if len(et_values) > 2 else 0
    else:
        persistence = 0
    
    # Calculate year-over-year variability
    unique_years = sorted(field_data['year'].unique())
    if len(unique_years) > 1:
        yearly_means = []
        for year in unique_years:
            year_data = field_data[field_data['year'] == year]['ET'].values
            if len(year_data) > 0:
                yearly_means.append(np.mean(year_data))
        
        if len(yearly_means) > 1:
            yearly_variability = np.std(yearly_means)
            yearly_cv = (yearly_variability / np.mean(yearly_means)) * 100 if np.mean(yearly_means) > 0 else 0
        else:
            yearly_variability = 0
            yearly_cv = 0
    else:
        yearly_variability = 0
        yearly_cv = 0
    
    # Detect trend direction
    if abs(trend_pct_per_period) < 0.5:
        trend_direction = "stable"
    elif trend_pct_per_period > 0:
        trend_direction = "increasing"
    else:
        trend_direction = "decreasing"
    
    # Classify persistence
    if persistence > 0.7:
        persistence_class = "high"
    elif persistence > 0.4:
        persistence_class = "moderate"
    else:
        persistence_class = "low"
    
    results.append({
        'field_id': field_id,
        'n_observations': len(et_values),
        'years_covered': len(unique_years),
        'mean_ET': mean_et,
        'std_ET': std_et,
        'cv_percent': cv,
        'trend_coefficient': trend_coef,
        'trend_percent_per_period': trend_pct_per_period,
        'trend_direction': trend_direction,
        'persistence_lag1': persistence,
        'persistence_class': persistence_class,
        'yearly_variability': yearly_variability,
        'yearly_cv_percent': yearly_cv,
        'min_ET': np.min(et_values),
        'max_ET': np.max(et_values),
        'range_ET': np.max(et_values) - np.min(et_values)
    })

# Create results DataFrame
results_df = pd.DataFrame(results)

print(f"\nAnalysis complete!")
print(f"Fields analyzed: {len(results_df)}")
print(f"\nTrend distribution:")
print(results_df['trend_direction'].value_counts())
print(f"\nPersistence distribution:")
print(results_df['persistence_class'].value_counts())

# Save results
results_df.to_csv(output_csv, index=False)
print(f"\nResults saved to: {output_csv}")

# Display summary statistics
print("\n" + "="*60)
print("SUMMARY STATISTICS")
print("="*60)
print(f"\nMean ET across fields: {results_df['mean_ET'].mean():.2f} ± {results_df['mean_ET'].std():.2f}")
print(f"Mean persistence (lag-1): {results_df['persistence_lag1'].mean():.3f}")
print(f"Mean trend (%/period): {results_df['trend_percent_per_period'].mean():.3f}")
print(f"Mean coefficient of variation: {results_df['cv_percent'].mean():.2f}%")

print("\nTop 10 fields with strongest increasing trend:")
print(results_df.nlargest(10, 'trend_percent_per_period')[['field_id', 'trend_percent_per_period', 'mean_ET']])

print("\nTop 10 fields with strongest decreasing trend:")
print(results_df.nsmallest(10, 'trend_percent_per_period')[['field_id', 'trend_percent_per_period', 'mean_ET']])

print("\nTop 10 fields with highest persistence:")
print(results_df.nlargest(10, 'persistence_lag1')[['field_id', 'persistence_lag1', 'mean_ET']])
