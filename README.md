# ET Time Series Analysis

This project analyzes evapotranspiration (ET) time series data to identify persistence, trends, and forecasts for field-level data.

## Scripts

### 1. `et_timeseries.py`
Extracts pixel-level ET data from raster files for field boundaries.

**Output:** `pixel_level_ET_timeseries.csv`
- 1,007,300 pixel-level records
- 111 fields analyzed
- 100 time periods (months 04, 05, 06, 10 from 2000-2024)

### 2. `analyze_timeseries.py`
Basic statistical analysis of trends and persistence.

**Features:**
- Linear trend analysis
- Persistence calculation (lag-1 autocorrelation)
- Coefficient of variation
- Year-over-year variability
- Trend classification (increasing/decreasing/stable)
- Persistence classification (high/moderate/low)

**Output:** `field_persistence_trends.csv`

### 3. `analyze_advanced.py` ⭐ RECOMMENDED
Advanced time series analysis with forecasting using exponential smoothing.

**Features:**
- All features from `analyze_timeseries.py`
- 12-period ahead forecasting using Holt-Winters exponential smoothing
- Seasonality detection and strength measurement
- Anomaly detection
- Forecast uncertainty quantification
- Forecast trend analysis

**Outputs:**
- `field_advanced_analysis.csv` - Comprehensive trends, persistence, and forecasts
- `field_forecasts.csv` - Detailed 12-step forecasts for each field

**Key Results:**
- Mean ET: 61.37 ± 10.15
- Mean persistence: 0.025 (low - high variability)
- Mean seasonality strength: 0.805
- Forecast mean ET: 55.40 (expected -9.72% change)

### 4. `analyze_with_chronos.py` (Optional)
Uses Amazon's Chronos foundation model for advanced forecasting.

**Requires:** Python 3.10+ and additional dependencies

### 5. `analyze_with_timesfm.py` (Optional)
Uses Google's TimesFM foundation model for advanced forecasting.

**Requires:** Python 3.10+ and complex dependencies

## Installation

### Option 1: Basic Analysis (Python 3.9+)
```bash
# Create virtual environment
python3 -m venv et-env
source et-env/bin/activate

# Install dependencies
pip install numpy pandas geopandas rioxarray rasterio tqdm statsmodels
```

### Option 2: With Foundation Models (Python 3.10+)
```bash
# Create virtual environment with Python 3.10+
python3.12 -m venv et-env-312
source et-env-312/bin/activate

# Install dependencies
pip install numpy pandas tqdm statsmodels

# For Chronos model
pip install torch transformers
pip install git+https://github.com/amazon-science/chronos-forecasting.git

# For TimesFM model (complex dependencies)
pip install timesfm
```

## Usage

```bash
# Activate environment
source et-env/bin/activate  # or et-env-312/bin/activate

# Step 1: Extract ET data from rasters (one-time)
python et_timeseries.py

# Step 2: Basic statistical analysis
python analyze_timeseries.py

# Step 3: Advanced analysis with forecasting (RECOMMENDED)
python analyze_advanced.py

# Step 4 (Optional): Foundation model analysis
python analyze_with_chronos.py  # or analyze_with_timesfm.py
```

## Output Fields

### Trends and Persistence Analysis

- `field_id`: Field identifier
- `n_observations`: Number of time periods
- `mean_ET`: Average ET value
- `std_ET`: Standard deviation
- `cv_percent`: Coefficient of variation (%)
- `trend_coefficient`: Linear trend slope
- `trend_percent_per_period`: Trend as % change per period
- `trend_direction`: increasing/decreasing/stable
- `persistence_lag1`: Lag-1 autocorrelation
- `persistence_class`: high/moderate/low
- `yearly_variability`: Year-to-year variation
- `min_ET`, `max_ET`, `range_ET`: Range statistics

### TimesFM Additional Fields

- `anomaly_score`: Deviation from historical mean (in std devs)
- `anomaly_status`: normal/moderate_anomaly/high_anomaly
- `forecast_mean_ET`: Mean of forecasted values
- `forecast_trend`: Trend in forecast period

## Notes

- TimesFM requires significant computational resources and may take longer to run
- The statistical analysis (`analyze_timeseries.py`) provides reliable results without requiring TimesFM
- TimesFM is recommended for forecasting and anomaly detection use cases
