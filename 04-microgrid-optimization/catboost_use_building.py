import pandas as pd
from catboost import CatBoostRegressor
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import os
from datetime import datetime
from scipy.io import savemat

# === Импорт твоих расчётных функций ===
from all_categories_and_strategy_YEAR_analysis_Daniil import (
    calculation1, calculation2, calculation3, calculation4
)

# === 1. Загрузка сохраненной модели ===
model = CatBoostRegressor()
model.load_model('catboost_model.cbm')
print("Модель успешно загружена")

# === 2. Загрузка и подготовка новых данных ===
new_data_path = r'C:\Users\danii\Downloads\output.xlsx'
new_df = pd.read_excel(new_data_path, parse_dates=["Date"])

print(f"Загружено строк: {len(new_df)}")
print(f"Колонки: {new_df.columns.tolist()}")

new_df["Date"] = pd.to_datetime(new_df["Date"])

# признаки
features = [
    "hour", "dayofweek", "month", "dayofyear", "is_weekend",
    "lag_1h", "lag_24h", "lag_168h",
    "rolling_mean_3h", "rolling_mean_24h", "rolling_mean_168h"
]

missing_features = set(features) - set(new_df.columns)
if missing_features:
    raise ValueError(f"Отсутствуют признаки: {missing_features}")

new_df_clean = new_df.dropna()
if len(new_df_clean) < len(new_df):
    print(f"Удалено {len(new_df) - len(new_df_clean)} строк с пропусками")

# === 3. Подготовка признаков ===
X_new = new_df_clean[features]

# === 4. Прогноз ===
print("Выполняется прогноз...")
predictions = model.predict(X_new)

# === 5. Сохраняем результат ===
results_df = new_df_clean[['Date']].copy()
results_df['predicted_load'] = predictions
results_df.rename(columns={'Date': 'time'}, inplace=True)

output_path = "new_data_forecast.xlsx"
results_df.to_excel(output_path, index=False)
print(f"Прогноз сохранен в: {output_path}")

# === 6. Добавляем ФАКТ и ГОД ===
if 'target' in new_df_clean.columns:
    y_true = new_df_clean['target'].values
elif 'Electricity load (kW)' in new_df_clean.columns:
    y_true = new_df_clean['Electricity load (kW)'].values
else:
    raise ValueError("Нет столбца с фактическим потреблением.")

results_df['fact'] = y_true
results_df['year'] = results_df['time'].dt.year

# === 7. Загрузка настроек и цен ===
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

years_in_data = sorted(results_df["year"].unique())
hourly_prices_full = []

for y in years_in_data:
    tmp = hourly_prices.copy()
    tmp.index = tmp.index.map(lambda dt: safe_replace_year(dt, y))
    hourly_prices_full.append(tmp)

hourly_prices = pd.concat(hourly_prices_full).sort_index()

allowed_keys = ['cost_cat1_rub', 'cost_cat22_rub', 'cost_cat23_rub', 'cost_cat3_rub', 'cost_cat4_rub']

# === 8. Расчет оптимальной категории по месяцам ===
all_results = []
results_df['month'] = results_df['time'].dt.month
year_month_pairs = sorted(results_df[['year', 'month']].drop_duplicates().values.tolist())

for (year, month) in year_month_pairs:
    month_data = results_df[(results_df['year'] == year) & (results_df['month'] == month)]
    res_fact = month_data[['time', 'fact']].rename(columns={'fact': 'consumption'}).set_index('time')
    res_forecast = month_data[['time', 'predicted_load']].rename(columns={'predicted_load': 'consumption'}).set_index('time')

    if res_fact.empty:
        continue

    start, end = res_fact.index.min(), res_fact.index.max()
    mask_prices = (hourly_prices.index >= start) & (hourly_prices.index <= end)
    if hourly_prices.loc[mask_prices].empty:
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
        "fact_optimal_cost": optimal_f.value,
        "forecast_optimal_cat": optimal_p.name,
        "forecast_optimal_cost": optimal_p.value,
        "delta_rub": optimal_p.value - optimal_f.value
    })


# === 9. Создание итоговой таблицы ===
df_summary = pd.DataFrame(all_results)
summary_path = os.path.join(base_path, "summary_categories_comparison_by_month.xlsx")
df_summary.to_excel(summary_path, index=False)


# === 10. Подсчёт совпадений и расхождений по категории ===
n_total = len(df_summary)
n_match = (df_summary['fact_optimal_cat'] == df_summary['forecast_optimal_cat']).sum()
n_diff = n_total - n_match

print(f"\nВсего месяцев: {n_total}")
print(f"Совпало по категории: {n_match}")
print(f"Отличаются: {n_diff}")
print(f"Процент совпадений: {n_match / n_total * 100:.2f}%")


# === 11. Улучшенные графики по годам ===

years = sorted([y for y in results_df['year'].unique() if 2014 <= y <= 2016])

if years:
    ncols = 3
    nrows = (len(years) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows))
    axes = axes.flatten()

    for i, year in enumerate(years):
        year_data = results_df[results_df['year'] == year]
        res_fact = year_data.set_index("time")["fact"]
        res_forecast = year_data.set_index("time")["predicted_load"]

        ax = axes[i]

        ax.plot(res_fact.index, res_fact.values, label="Факт", linewidth=2.5)
        ax.plot(res_forecast.index, res_forecast.values, label="Прогноз", linewidth=2.5, linestyle="--")

        ax.set_title(f"Year {year}", fontsize=14)
        ax.tick_params(axis='x', rotation=30, labelsize=12)
        ax.tick_params(axis='y', labelsize=12)

        # Рамка
        for side in ["bottom", "top", "left", "right"]:
            ax.spines[side].set_linewidth(1.5)

        # Сетка (можно выключить)
        ax.grid(True, alpha=0.3)

        if i == 0:
            ax.legend(fontsize=12)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.show()


# === 12. Метрики ===
mae = mean_absolute_error(results_df["fact"], results_df["predicted_load"])
rmse = np.sqrt(mean_squared_error(results_df["fact"], results_df["predicted_load"]))
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")


# === 13. Сохранение всех результатов в MATLAB (.mat) ===
matlab_data = {
    'results_df': results_df.to_dict("list"),
    'df_summary': df_summary.to_dict("list"),
    'MAE': mae,
    'RMSE': rmse,
}
savemat("results_for_matlab.mat", matlab_data)

print("MATLAB файл сохранён: results_for_matlab.mat")
print("\nГотово!")
