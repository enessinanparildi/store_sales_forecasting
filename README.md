# Store Sales Forecasting

A time series forecasting project for predicting daily store sales using CatBoost regression with advanced feature engineering and temporal analysis.

## Overview

This project implements a machine learning pipeline to forecast sales across multiple retail stores. It leverages historical sales data, promotional information, and store characteristics to predict future sales with high accuracy. The model uses gradient boosting (CatBoost) combined with sophisticated time series feature engineering.

## Features

- **Comprehensive Data Processing**
  - Automatic handling of missing dates and store closures
  - Cyclical promotion detection and spot promotion tracking
  - Competition distance analysis and preprocessing

- **Advanced Feature Engineering**
  - Multi-horizon lag features (1, 7, 14, 21, 28, 35, 364 days)
  - Exponentially weighted moving averages and standard deviations
  - Rolling statistics with multiple window sizes
  - Fourier transform features for capturing weekly seasonality
  - Momentum indicators and difference features

- **Exploratory Analysis**
  - ACF/PACF plots for autocorrelation analysis
  - Seasonal decomposition with heteroskedasticity testing
  - Sales distribution visualizations
  - Day-of-week heatmaps
  - Promotional activity trends

- **Model Performance**
  - RMSE: ~762 (significantly better than naive baselines)
  - MAE: ~482
  - Comparison with lag-7, lag-1, and lag-14 naive forecasts
  - Per-store error analysis

## Installation

```bash
pip install pandas numpy catboost icecream matplotlib seaborn statsmodels scikit-learn
```

## Data Structure

The project expects three CSV files in the `data/` directory:

1. **dataset.csv** - Historical sales data
   - Store, Date, Sales, Customers

2. **promotions.csv** - Spot promotion dates
   - Store, Date

3. **stores.csv** - Store characteristics
   - Store, Assortment, CompetitionDistance, PromoSinceWeek, PromoSinceYear, PromoInterval

## Usage

### Running the Main Script

```bash
python main.py
```

This will:
- Process and merge all datasets
- Generate exploratory plots (ACF/PACF, histograms, seasonal decomposition)
- Engineer features
- Train the CatBoost model
- Output validation metrics and predictions

### Using the Jupyter Notebook

```bash
jupyter notebook notebook_forecast.ipynb
```

The notebook provides an interactive environment for:
- Step-by-step data exploration
- Visualization of intermediate results
- Model training and evaluation
- Hyperparameter tuning experiments

## Model Architecture

### CatBoost Regressor Configuration

```python
{
    "loss_function": "RMSE",
    "iterations": 7000,
    "learning_rate": 0.04-0.06,
    "depth": 8,
    "l2_leaf_reg": 3.0,
    "subsample": 0.8,
    "bootstrap_type": "Bernoulli",
    "od_type": "Iter",
    "od_wait": 200
}
```

### Key Features Used

- **Temporal**: weekday, is_weekend, day, month, year
- **Lag Features**: Sales and Customer counts at various lags
- **Rolling Statistics**: 7, 28, 56-day windows
- **Exponential Smoothing**: 7 and 28-day EWMA/EWSTD
- **Promotions**: spot_promo_flag, cyclical_promo_flag, promo × weekend interaction
- **Store Characteristics**: assortment_code, CompetitionDistance
- **Seasonality**: Fourier transform features (sin/cos)

## Results

### Model Performance (Validation Set)

| Metric | Model | Naive (Lag-7) | Naive (Lag-1) | Naive (Lag-14) |
|--------|-------|---------------|---------------|----------------|
| RMSE   | 761.7 | 3266.8        | 4785.6        | 2594.5         |
| MAE    | 481.7 | 2175.0        | 3100.4        | 1425.3         |

The model achieves **76.7% improvement** in RMSE over the lag-7 baseline and **70.6% improvement** over the lag-14 baseline.

### Sample Store Performance

RMSE varies by store characteristics, ranging from ~370 to ~960 across different store types.

## Project Structure

```
store_sales_forecasting/
├── data/
│   ├── dataset.csv
│   ├── promotions.csv
│   └── stores.csv
├── main.py                          # Main training script
├── notebook_forecast.ipynb          # Interactive notebook
├── README.md                        # This file
└── outputs/                         # Generated plots and results
    ├── acf_pacf_plot_*.png
    ├── sales_histogram_*.png
    ├── non_log_seasonal_decompose_*.png
    ├── plot_*.png
    ├── CompetitionDistance_histogram.png
    ├── MeanSalesbyStore.png
    └── daily_promo_active_mean.png
```

## Technical Highlights

1. **Log Transformation**: Stabilizes variance and reduces heteroskedasticity in sales data
2. **ARCH Testing**: Validates presence/absence of conditional heteroskedasticity in residuals
3. **Categorical Encoding**: Efficient handling of Store and Assortment categories
4. **Missing Data Strategy**: Strategic filling using forward-fill for store IDs and maximum values for competition distance
5. **Train/Validation Split**: 90/10 split based on temporal ordering

## Future Improvements

- [ ] Implement cross-validation with time-based folds
- [ ] Add external features (weather, holidays, economic indicators)
- [ ] Experiment with ensemble methods (stacking multiple models)
- [ ] Implement Bayesian hyperparameter optimization
- [ ] Add confidence intervals for predictions
- [ ] Create interactive dashboard for predictions

## Notes

- The model uses GPU acceleration when available (set `task_type="GPU"` in main.py)
- Early stopping is implemented with 200-iteration patience
- All lag features are back-filled to avoid look-ahead bias
- The empty_store_flag tracks store closure periods
**Note**: Make sure your GPU drivers and CUDA are properly installed if you want to use GPU acceleration for faster training.
