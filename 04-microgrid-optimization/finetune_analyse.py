import pandas as pd
from catboost import CatBoostRegressor
import matplotlib.pyplot as plt
from scipy.io import savemat
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import os
from datetime import datetime
from sklearn.model_selection import train_test_split

# === Import your calculation functions ===
from all_categories_and_strategy_YEAR_analysis_Daniil import (
    calculation1, calculation2, calculation3, calculation4
)

# === 1. Load the saved model ===
model = CatBoostRegressor()
model.load_model('catboost_model.cbm')
print("Model loaded successfully")

# === 2. Load and prepare new data ===
new_data_path = r'C:\Users\danii\Downloads\output.xlsx'
new_df = pd.read_excel(new_data_path, parse_dates=["Date"])
print(f"Loaded rows: {len(new_df)}")
print(f"Columns: {new_df.columns.tolist()}")

new_df["Date"] = pd.to_datetime(new_df["Date"])

features = [
    "hour", "dayofweek", "month", "dayofyear", "is_weekend",
    "lag_1h", "lag_24h", "lag_168h",
    "rolling_mean_3h", "rolling_mean_24h", "rolling_mean_168h"
]

missing_features = set(features) - set(new_df.columns)
if missing_features:
    raise ValueError(f"Missing features: {missing_features}")

new_df_clean = new_df.dropna()
if len(new_df_clean) < len(new_df):
    print(f"Removed {len(new_df) - len(new_df_clean)} rows with NaN values")

X_new = new_df_clean[features]

if 'target' in new_df_clean.columns:
    y_true = new_df_clean['target'].values
elif 'Electricity load (kW)' in new_df_clean.columns:
    y_true = new_df_clean['Electricity load (kW)'].values
else:
    raise ValueError("Missing target column ('target' or 'Electricity load (kW)')")

# === 3. Load external data (prices and settings) ===
base_path = os.path.dirname(os.path.abspath(__file__))
user_data = pd.read_excel(os.path.join(base_path, "настройки.xlsx"), header=0)
category_contract = user_data.iloc[0, 0]
v_level = user_data.iloc[0, 1]
s_level = user_data.iloc[0, 2]
year_user = user_data.iloc[0, 3]
company = user_data.iloc[0, 5]

prices1 = pd.read_excel(os.path.join(base_path, "Почасовые цены1.xlsx"), parse_dates=["time"])
prices2 = pd.read_excel(os.path.join(base_path, "Почасовые цены2.xlsx"), parse_dates=["time"])
hourly_prices = pd.concat([prices1, prices2]).set_index("time").sort_index()

def safe_replace_year(dt, new_year):
    try:
        return dt.replace(year=int(new_year))
    except ValueError:
        if dt.month == 2 and dt.day == 29:
            return dt.replace(year=int(new_year), day=28)
        return dt

years_in_data = sorted(new_df_clean["Date"].dt.year.unique())
hourly_prices_full = []
for y in years_in_data:
    tmp = hourly_prices.copy()
    tmp.index = tmp.index.map(lambda dt: safe_replace_year(dt, y))
    hourly_prices_full.append(tmp)
hourly_prices = pd.concat(hourly_prices_full).sort_index()

allowed_keys = ['cost_cat1_rub', 'cost_cat22_rub', 'cost_cat23_rub', 'cost_cat3_rub', 'cost_cat4_rub']

# === 4. Dependency of category match percentage on fine-tune data ratio ===
print("\n=== 📊 Analyzing dependency of category match percentage on fine-tuning data ratio ===")

test_sizes = np.arange(0.1, 0.95, 0.05)
fine_tune_ratios = 1 - test_sizes
category_match_percent = []
mae_list = []
rmse_list = []

for ts in test_sizes:
    print(f"\n--- test_size = {ts:.2f} (fine-tune = {1 - ts:.2f}) ---")

    X_ft, X_forecast, y_ft, y_forecast = train_test_split(
        X_new, y_true,
        test_size=ts,
        random_state=42,
        shuffle=False
    )

    X_train_ft, X_val_ft, y_train_ft, y_val_ft = train_test_split(
        X_ft, y_ft, test_size=0.2, random_state=42, shuffle=False
    )

    model_tmp = CatBoostRegressor()
    model_tmp.load_model("catboost_model.cbm")

    model_tmp.fit(
        X_train_ft,
        y_train_ft,
        eval_set=(X_val_ft, y_val_ft),
        init_model=model_tmp,
        early_stopping_rounds=50,
        use_best_model=True,
        verbose=False
    )

    preds_forecast = model_tmp.predict(X_forecast)
    preds_val = model_tmp.predict(X_val_ft)

    mae = mean_absolute_error(y_val_ft, preds_val)
    rmse = np.sqrt(mean_squared_error(y_val_ft, preds_val))

    mae_list.append(mae)
    rmse_list.append(rmse)

    df_tmp = pd.DataFrame({
        "time": new_df_clean.iloc[X_forecast.index]["Date"].values,
        "predicted_load": preds_forecast,
        "actual_load": y_forecast
    })
    df_tmp["year"] = pd.to_datetime(df_tmp["time"]).dt.year
    df_tmp["month"] = pd.to_datetime(df_tmp["time"]).dt.month

    all_results = []
    for (year, month) in sorted(df_tmp[["year", "month"]].drop_duplicates().values.tolist()):
        month_data = df_tmp[(df_tmp["year"] == year) & (df_tmp["month"] == month)]
        if len(month_data) < 10:
            continue

        res_fact = month_data[["time", "actual_load"]].rename(columns={'actual_load': 'consumption'}).set_index("time")
        res_forecast = month_data[["time", "predicted_load"]].rename(columns={'predicted_load': 'consumption'}).set_index("time")

        start, end = res_fact.index.min(), res_fact.index.max()
        mask_prices = (hourly_prices.index >= start) & (hourly_prices.index <= end)
        prices_for_period = hourly_prices.loc[mask_prices]

        if prices_for_period.empty:
            continue

        df1_f = calculation1(hourly_prices, res_fact, company, category_contract, v_level, s_level, year_user, year)
        df2_f = calculation2(hourly_prices, res_fact, company, category_contract, v_level, s_level, year_user, year)
        df3_f = calculation3(hourly_prices, res_fact, company, category_contract, v_level, s_level, year_user, year)
        df4_f = calculation4(hourly_prices, res_fact, company, category_contract, v_level, s_level, year_user, year)
        prices_f = pd.concat([df1_f, df2_f, df3_f, df4_f])
        optimal_f = prices_f.loc[allowed_keys].iloc[prices_f.loc[allowed_keys]['value'].argmin()]

        df1_p = calculation1(hourly_prices, res_forecast, company, category_contract, v_level, s_level, year_user, year)
        df2_p = calculation2(hourly_prices, res_forecast, company, category_contract, v_level, s_level, year_user, year)
        df3_p = calculation3(hourly_prices, res_forecast, company, category_contract, v_level, s_level, year_user, year)
        df4_p = calculation4(hourly_prices, res_forecast, company, category_contract, v_level, s_level, year_user, year)
        prices_p = pd.concat([df1_p, df2_p, df3_p, df4_p])
        optimal_p = prices_p.loc[allowed_keys].iloc[prices_p.loc[allowed_keys]['value'].argmin()]

        all_results.append({
            "year": year,
            "month": month,
            "fact_optimal_cat": optimal_f.name,
            "forecast_optimal_cat": optimal_p.name
        })

    df_res = pd.DataFrame(all_results)
    if not df_res.empty:
        n_total = len(df_res)
        n_match = (df_res["fact_optimal_cat"] == df_res["forecast_optimal_cat"]).sum()
        percent_match = n_match / n_total * 100
    else:
        percent_match = np.nan

    category_match_percent.append(percent_match)
    print(f"→ Category match: {percent_match:.2f}% | MAE={mae:.3f} | RMSE={rmse:.3f}")


# === 5. Improved Plot ===

plt.rcParams.update({
    'font.size': 18,
    'axes.titlesize': 22,
    'axes.labelsize': 20,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'legend.fontsize': 18
})

fig, ax1 = plt.subplots(figsize=(14, 8))

# --- Add boxing line (рамка)
for spine in ax1.spines.values():
    spine.set_linewidth(2)

ax1.plot(
    fine_tune_ratios * 100,
    category_match_percent,
    marker='o',
    markersize=10,
    color='green',
    linewidth=3,
    label="% category matches"
)

ax1.set_xlabel("Data share for fine-tuning (%)")
ax1.set_ylabel("Category Match (%)", color='green')
ax1.tick_params(axis='y', labelcolor='green')

ax1.grid(True, linestyle='--', alpha=0.5)

ax2 = ax1.twinx()

# Boxing line for second axis
for spine in ax2.spines.values():
    spine.set_linewidth(2)

ax2.plot(
    fine_tune_ratios * 100,
    mae_list,
    marker='s',
    markersize=10,
    linestyle='--',
    linewidth=3,
    color='blue',
    label="MAE"
)

ax2.plot(
    fine_tune_ratios * 100,
    rmse_list,
    marker='^',
    markersize=10,
    linestyle=':',
    linewidth=3,
    color='red',
    label="RMSE"
)

ax2.set_ylabel("Metrics (MAE, RMSE)", color='gray')
ax2.tick_params(axis='y', labelcolor='gray')

ax1.set_title("The influence of the amount of data for additional training\n on the accuracy of categories and model errors")

ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

plt.tight_layout()
plt.show()

print("\n=== Analysis completed ===")

results_dict = {
    "fine_tune_ratios": fine_tune_ratios,
    "category_match_percent": np.array(category_match_percent),
    "mae_list": np.array(mae_list),
    "rmse_list": np.array(rmse_list),
}

savemat("fine_tune_analysis_results.mat", results_dict)
print("MATLAB .mat file saved: fine_tune_analysis_results.mat")