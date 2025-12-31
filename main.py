import pandas as pd
import numpy as np
from catboost import CatBoostRegressor, Pool
from icecream import ic
import matplotlib
import matplotlib.pyplot as plt
import os

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.stats.diagnostic import het_arch

from sklearn.metrics import root_mean_squared_error, mean_absolute_error
import seaborn as sns



os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
matplotlib.use("Agg")  # disables GUI plotting


def question_answers():
    dataset_df = pd.read_csv("./data/dataset.csv")
    promotions_df = pd.read_csv("./data/promotions.csv")
    stores_df = pd.read_csv("./data/stores.csv")

    # First Question
    emptypromosinceweek = len(stores_df['PromoSinceWeek'][stores_df['PromoSinceWeek'].isna()])
    emptypromosinceyear = len(stores_df['PromoSinceYear'][stores_df['PromoSinceYear'].isna()])
    emptyPromoInterval = len(stores_df['PromoInterval'][stores_df['PromoInterval'].isna()])
    emptycomptdistance = len(stores_df['CompetitionDistance'][stores_df['CompetitionDistance'].isna()])



    # Second Question day collapsed
    result_df_total_sales_customers = dataset_df.groupby('Store')[['Sales', 'Customers']].sum().reset_index()
    print(len(result_df_total_sales_customers))
    print(len(stores_df))
    merged_data = pd.merge(result_df_total_sales_customers, stores_df, on='Store', how='left')
    merged_data = merged_data[['Store', "Sales", "Customers", "Assortment"]].copy()
    mean_by_assortment = merged_data.groupby('Assortment')[['Sales', 'Customers']].mean().reset_index()
    mean_by_assortment

    # Second Question non-day collapsed
    merged_data_stores = pd.merge(dataset_df, stores_df, on='Store', how='left')
    merged_data_stores = merged_data_stores[['Store', "Sales", "Date", "Customers", "Assortment"]].copy()
    mean_by_assortment_each_day = merged_data_stores.groupby(by = ["Assortment", "Date"])[['Sales', 'Customers']].mean().reset_index()



    # Third Question
    dataset_df['Date'] = pd.to_datetime(dataset_df['Date'])
    dataset_df_2014 = dataset_df[dataset_df['Date'].dt.year == 2014]
    dataset_df_2014_grouped = dataset_df_2014.groupby('Store')[['Sales', 'Customers']].sum().reset_index()
    dataset_df_2014_grouped_sorted = dataset_df_2014_grouped.sort_values(by='Sales', ascending=False)
    print(dataset_df_2014_grouped_sorted.iloc[0]['Store'])


    # Fourth Question
    dataset_df['Date'] = pd.to_datetime(dataset_df['Date'])
    stores_df_filtered = stores_df[
        (stores_df['CompetitionDistance'] < 1000) &
        (stores_df['PromoInterval'] == 'Jan,Apr,Jul,Oct')
    ]
    dataset_df_grouped = dataset_df.groupby(["Store", dataset_df['Date'].dt.to_period('M')])['Sales'].mean().reset_index()
    merged_data = pd.merge(stores_df_filtered, dataset_df_grouped,  on='Store', how='left')
    merged_data = merged_data[['Store', 'Date', 'Sales']].copy()
    final_grouped_data = merged_data.groupby("Date")["Sales"].mean().reset_index()


# TODO: plat sales distribittion raw and log)
# TODO: Histogram of CompetitionDistance (log scale); relation to average sales (scatter with trend)
#

# **ACF**
# Measures correlation between the series and its lagged versions **including indirect effects**.

# ## **PACF**
# Measures **direct** correlation between the series and its lag at k **after removing effects of all intermediate lags**.
def plot_acf_pacf(df, store_id, max_lag=60):
    df = df[df["empty_store_flag"] == 0]
    s = (df.loc[df["Store"] == store_id, ["Date", "Sales"]].sort_values("Date").set_index("Date")["Sales"])

    s = s.asfreq("D")
    s_clean = s.dropna()
    fig, axes = plt.subplots(1, 2, figsize=(12,4))
    plot_acf(s_clean, lags=max_lag, ax=axes[0])
    plot_pacf(s_clean, lags=max_lag, ax=axes[1], method="ywm")
    axes[0].set_title("ACF")
    axes[1].set_title("PACF")
    fig.savefig(f"acf_pacf_plot_{store_id}.png", dpi=300, bbox_inches="tight")

def plot_sales_hist(df, store_id):
    df = df[df["empty_store_flag"] == 0]
    s = df.loc[df["Store"] == store_id, ["Date", "Sales"]]

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(s['Sales'], bins=40, kde=False, color='skyblue', ax=ax)
    ax.set_title(f'Sales Distribution - Store {store_id}')
    ax.set_xlabel('Sales')
    ax.set_ylabel('Frequency')

    fig.savefig(f"sales_histogram_{store_id}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

def plot_comp_dist_hist(df):
    df = df[df["empty_store_flag"] == 0]
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(df['CompetitionDistance'], bins=40, kde=False, color='skyblue', ax=ax)
    ax.set_title(f'CompetitionDistance')
    ax.set_xlabel('CompetitionDistance')
    ax.set_ylabel('Frequency')

    fig.savefig(f"CompetitionDistance_histogram.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_seasonal_decompose(df, store_id):
    df = df[df["empty_store_flag"] == 0]
    s = df.loc[df["Store"] == store_id, ["Date", "Sales"]]
    s = s['Sales'].astype(float)

    #s = np.log1p(s['Sales'].astype(float))
    trs = seasonal_decompose(s, model='additive', period=7)

    # Check for heteroskedasticity
    resid = trs.resid
    resid = resid[~resid.isna()]
    test_stat, p_value, _, _ = het_arch(resid)

    # if < 0.05 Residuals are heteroskedastic (ARCH effect exists)
    ic(store_id)
    print("ARCH test statistic:", test_stat)
    print("p-value:", p_value)

    fig = trs.plot()
    fig.set_size_inches(12, 8)
    fig.tight_layout()
    fig.savefig(f"non_log_seasonal_decompose_{store_id}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

def weekly_heatmap(df):
    tmp = df[df["empty_store_flag"] == 0].copy()
    heat = tmp.groupby(["Store", "weekday"])["Sales"].mean().unstack("weekday")

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(heat.sample(60), cmap="viridis", ax=ax)
    ax.set_title("Mean Sales by Store × Day-of-Week (sample)")

    fig.savefig("MeanSalesbyStore.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

def plot_day_promo_dist(df):
    df = df[df["empty_store_flag"] == 0].copy()
    df["PromoActive"] = ((df.get("spot_promo_flag", 0).astype(int) == 1) | (df.get("cyclical_promo_flag", 0).astype(int) == 1)).astype(int)
    df_grouped_promo_mean = df.groupby("Date")["PromoActive"].mean().reset_index()
    plt.figure(figsize=(12, 5))
    plt.plot(df_grouped_promo_mean["Date"], df_grouped_promo_mean["PromoActive"])
    plt.xlabel("Date")
    plt.ylabel("Mean PromoActive")
    plt.title("Daily Mean PromoActive")
    # Save
    plt.savefig("daily_promo_active_mean.png", dpi=300, bbox_inches="tight")
    plt.close()

# Modelling
if __name__ == "__main__":
    dataset_df = pd.read_csv("./data/dataset.csv")
    promotions_df = pd.read_csv("./data/promotions.csv")
    stores_df = pd.read_csv("./data/stores.csv")

    dataset_df['Date'] = pd.to_datetime(dataset_df['Date'])

    # Get missing dates for each store
    def missing_dates(dates):
        full = pd.date_range(dates.min(), dates.max(), freq='D')
        return full.difference(dates)

    missing = dataset_df.groupby('Store')['Date'].apply(missing_dates)

    # Fill missing dates for each store and create empty store time flag
    def fill_dates(dates):
        full_index = pd.date_range(start=dates["Date"].min(), end=dates["Date"].max(), freq='D')
        dates = dates.set_index('Date').reindex(full_index)
        return dates

    dataset_df = dataset_df.groupby('Store', group_keys=False).apply(fill_dates).reset_index().rename(columns={'index': 'Date'})
    dataset_df['Store'] = dataset_df['Store'].ffill()
    dataset_df['empty_store_flag'] = 0
    dataset_df.loc[dataset_df['Sales'].isna(), 'empty_store_flag'] = 1
    dataset_df["Sales"] = dataset_df["Sales"].fillna(0)
    dataset_df["Customers"] = dataset_df["Customers"].fillna(0)


    # Get and merge daily spot promo flag
    promotions_df['spot_promo_flag'] = 1
    promotions_df['Date'] = pd.to_datetime(promotions_df['Date'])
    merged_data = pd.merge(dataset_df, promotions_df, on=['Store', "Date"], how='left')
    merged_data['spot_promo_flag'] = merged_data['spot_promo_flag'].fillna(0)
    merged_data_all = pd.merge(merged_data, stores_df, on='Store', how = 'left')


    # Calculate cyclical promo flag
    def check_cyclical_promo(row):
        month_map = {
            'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4,
            'May': 5, 'Jun': 6, 'Jul': 7, 'Aug': 8,
            'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
        }

        if pd.isna(row['PromoSinceYear']) or row['PromoSinceWeek'] == 0:
            return 0

        if pd.isna(row['PromoInterval']):
            return 0

        try:
            promo_start_week = int(row['PromoSinceWeek'])
            promo_start_year = int(row['PromoSinceYear'])
        except:
            return 0

        current_year = row['Date'].year
        current_week = row['Date'].isocalendar()[1]
        current_month = row['Date'].month

        if current_year < promo_start_year:
            return 0
        elif current_year == promo_start_year and current_week < promo_start_week:
            return 0
        else:
            promo_months = row['PromoInterval'].split(',')
            promo_month_nums = [month_map.get(m.strip(), 0) for m in promo_months]

            if current_month in promo_month_nums:
                return 1
            else:
                return 0

    merged_data_all['cyclical_promo_flag'] = merged_data_all.apply(check_cyclical_promo, axis=1)

    # Preprocess competition distance
    merged_data_all['empty_comp_distance'] = 0
    merged_data_all.loc[merged_data_all['CompetitionDistance'].isna(), 'empty_comp_distance'] = 1
    merged_data_all['CompetitionDistance'] = merged_data_all['CompetitionDistance'].fillna(max(merged_data_all['CompetitionDistance']))
    merged_data_all['CompetitionDistance'] = np.log1p(merged_data_all['CompetitionDistance'])


    features = merged_data_all[['Store', 'Date', "Customers", 'Sales', 'spot_promo_flag', 'CompetitionDistance',
                                'cyclical_promo_flag', 'empty_comp_distance', "empty_store_flag"]]

    # Run plotting utilities
    stores = features['Store'].unique()
    for s in stores[::180]:
        tmp = features[features['Store'] == s].sort_values('Date')
        tmp_ = tmp[tmp["empty_store_flag"] == 0]
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(tmp_['Date'], tmp_['Sales'])
        ax.set_title(f"Store {s} – Sales Over Time")
        ax.set_xlabel("Date")
        ax.set_ylabel("Sales")

        fig.savefig(f"plot_{s}.png", dpi=300, bbox_inches="tight")
        plot_acf_pacf(features, s, max_lag=60)
        plot_sales_hist(features, s)
        plot_seasonal_decompose(features, s)

    plot_comp_dist_hist(features)

    # Run feature engineering steps
    features = features.sort_values(["Store","Date"])

    features['assortment_code'] = merged_data_all['Assortment'].astype('category').cat.codes
    features['Store'] = features['Store'].astype(str)  # safest for CatBoost categorical

    features["weekday"] = features["Date"].dt.weekday
    features["is_weekend"] = (features["weekday"] >= 5).astype(int)
    features["day"] = features["Date"].dt.day
    features["month"] = features["Date"].dt.month
    features["year"] = features["Date"].dt.year

    weekly_heatmap(features)
    plot_day_promo_dist(features)


    # Sales lag features
    for L in [1,2,3,4,5,6, 7, 14, 21,  28, 35, 364]:
        features[f"Sales_lag_{L}"] = features.groupby("Store")["Sales"].shift(L)
        features[f"Sales_lag_{L}"] = features[f"Sales_lag_{L}"].bfill()

        features[f"Customer_lag_{L}"] = features.groupby("Store")["Customers"].shift(L)
        features[f"Customer_lag_{L}"] = features[f"Customer_lag_{L}"].bfill()

        features[f"is_open_lag_{L}"] = features.groupby("Store")["empty_store_flag"].shift(L)
        features[f"is_open_lag_{L}"] = features[f"is_open_lag_{L}"].bfill()

    features["Sales_lag_diff_1"] =  features["Sales_lag_1"] -  features["Sales_lag_7"]
    features["Sales_lag_diff_2"] =  features["Sales_lag_7"] -  features["Sales_lag_14"]

    features["spot_promo_x_weekend"] = features["spot_promo_flag"] * features["is_weekend"]

    # store_collapsed = features.groupby("Date")["Sales"].mean().reset_index()
    # for k in [1,2,3,4,5,6, 7, 14, 21,  28, 35, 364]:
    #     store_collapsed[f"Sales_mean_lag_{k}"] = store_collapsed["Sales"].shift(k)
    # features_mean_lagged_version = features.merge(store_collapsed, on="Date", how="left")

    g = features.groupby("Store")
    features["t"] = g["Date"].transform(lambda x: (x - x.min()).dt.days)
    for k in [1, 2, 3]:
        features[f"sin_wk_{k}"] = np.sin(2*np.pi *k * features["t"] / 7)
        features[f"cos_wk_{k}"] = np.cos(2*np.pi *k * features["t"] / 7)

    # An exponentially weighted (EW) mean is a moving average where recent observations
    # get more weight and older observations get exponentially decreasing weight.
    features["ewma_7"] = features.groupby("Store")["Sales"].transform(lambda s: s.shift(1).ewm(span=7, adjust=False, min_periods=3).mean())
    features["ewma_28"] = features.groupby("Store")["Sales"].transform(lambda s: s.shift(1).ewm(span=28, adjust=False, min_periods=7).mean())

    # Exponentially weighted std (volatility), e.g., 7-day
    features["ewstd_7"] = features.groupby("Store")["Sales"].transform(lambda s: s.shift(1).ewm(span=7, adjust=False, min_periods=3).std())
    # Compute rolling averages and rolling standard deviations over 7-28 days, use at least 3 valid days
    for W in [7, 28, 56]:
        features[f"Sales_rollmean_{W}"] = features.groupby("Store")["Sales"].transform(lambda s: s.shift(1).rolling(W, min_periods=3).mean())
        features[f"Sales_rollstd_{W}"] = features.groupby("Store")["Sales"].transform(lambda s: s.shift(1).rolling(W, min_periods=3).std())

    #a scale‑free momentum signal. By dividing “yesterday” by a recent average,
    # we tell the model whether the latest level is unusually high or low for this store right now,
    # without caring about the store’s absolute scale
    features["lag1_over_roll7"] = features["Sales_lag_1"] / (features["Sales_rollmean_7"] + 1e-6)

    feature_cols = ['Store', 'spot_promo_flag', 'CompetitionDistance', "empty_store_flag",
           'cyclical_promo_flag', 'empty_comp_distance', 'assortment_code', 'weekday',
           'is_weekend', 'day', 'month', 'year', 'Sales_lag_1', 'Customer_lag_1', 'Sales_lag_2',
            'Customer_lag_2', 'Sales_lag_3', 'Customer_lag_3', 'Sales_lag_4',
            'Customer_lag_4', 'Sales_lag_5', 'Customer_lag_5', 'Sales_lag_6',
            'Customer_lag_6', 'Sales_lag_7', 'Customer_lag_7', 'Sales_lag_14',
            'Customer_lag_14', 'Sales_lag_21', 'Customer_lag_21', 'Sales_lag_28',
            'Customer_lag_28', 'Sales_lag_35', 'Customer_lag_35', 'Sales_lag_364',
            'Customer_lag_364', "spot_promo_x_weekend", "ewma_7", "ewma_28", "ewstd_7",
           'Sales_rollmean_7', 'Sales_rollstd_7', 'Sales_rollmean_28',
           'Sales_rollstd_28', 'Sales_rollmean_56', 'Sales_rollstd_56', "Sales_lag_diff_1", "Sales_lag_diff_2",
            'sin_wk_1', 'cos_wk_1', 'sin_wk_2', 'cos_wk_2', 'sin_wk_3', 'cos_wk_3', 'is_open_lag_1', 'is_open_lag_2', 'is_open_lag_3',
            'is_open_lag_4', 'is_open_lag_5', 'is_open_lag_6', 'is_open_lag_7',
            'is_open_lag_14', 'is_open_lag_21', 'is_open_lag_28', 'is_open_lag_35',
            'is_open_lag_364']


    cat_cols = ["Store","assortment_code"]
    cat_idxs = [feature_cols.index(c) for c in cat_cols if c in feature_cols]

    cutoff = features["Date"].quantile(0.9) # or a specific date
    train = features[(features["Date"] <= cutoff)].dropna(subset=feature_cols + ["Sales"])
    valid = features[(features["Date"] > cutoff)].dropna(subset=feature_cols + ["Sales"])

    X_tr, y_tr = train[feature_cols], train["Sales"]
    X_va, y_va = valid[feature_cols], valid["Sales"]

    # Log transform stabilizes variance, reduces heteroskedasticity
    # Convert multiplicative effects to additive
    y_tr = np.log1p(y_tr)
    y_va = np.log1p(y_va)

    ic(len(y_tr))
    ic(len(y_va))

    pool_tr = Pool(X_tr, y_tr, cat_features=cat_idxs)
    pool_va = Pool(X_va, y_va, cat_features=cat_idxs)

    model = CatBoostRegressor(
    loss_function="RMSE",
    iterations=7000,
    learning_rate=0.06,
    depth=8,
    l2_leaf_reg=3.0,
    subsample=0.8,
    random_seed=42,
    od_type="Iter",
    bootstrap_type='Bernoulli',
    od_wait=200,
    eval_metric="RMSE",
    task_type="GPU",
    verbose=200
    )
    model.fit(pool_tr, eval_set=pool_va)
    predictions_validation = model.predict(pool_va)

    # This takes too long
#    importance = model.get_feature_importance(data=pool_tr, type="LossFunctionChange")
#    fi = pd.DataFrame({
#        'feature': model.feature_names_,
#        'importance': importance
#    }).sort_values('importance', ascending=False)


    yhat = np.expm1(predictions_validation)
    y_og = np.expm1(y_va)
    valid['predictions'] = yhat

    model_rmse_error = root_mean_squared_error(yhat, y_og)
    ic(model_rmse_error)

    # MAE is ideal when we want a straightforward measure of average error magnitude without emphasizing large errors disproportionately.
    model_mae_error = mean_absolute_error(yhat, y_og)
    ic(model_mae_error)


    print("Per Store Error")
    stores = features['Store'].unique()
    for store_id in stores[::180]:
        yhatfilter = valid[valid['Store'] == str(store_id)]["predictions"]
        yogfilter =  valid[valid['Store'] == str(store_id)]["Sales"]
        model_rmse_error = root_mean_squared_error(yhatfilter, yogfilter)
        ic(store_id)
        ic(model_rmse_error)

    naive_rmse_error_lag7 = root_mean_squared_error(X_va['Sales_lag_7'], y_og)
    ic(naive_rmse_error_lag7)

    naive_mae_error_lag7 = mean_absolute_error(X_va['Sales_lag_7'], y_og)
    ic(naive_mae_error_lag7)

    naive_rmse_error_lag1 = root_mean_squared_error(X_va['Sales_lag_1'], y_og)
    ic(naive_rmse_error_lag1)

    naive_mae_error_lag1 = mean_absolute_error(X_va['Sales_lag_1'], y_og)
    ic(naive_mae_error_lag1)

    naive_rmse_error_lag14 = root_mean_squared_error(X_va['Sales_lag_14'], y_og)
    ic(naive_rmse_error_lag14)

    naive_mae_error_lag14 = mean_absolute_error(X_va['Sales_lag_14'], y_og)
    ic(naive_mae_error_lag14)

