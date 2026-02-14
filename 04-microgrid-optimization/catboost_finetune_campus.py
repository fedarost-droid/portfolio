import os
import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from catboost import CatBoostRegressor
from scipy.io import savemat
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

# === Импорт расчётных функций ===
from all_categories_and_strategy_YEAR_analysis_Daniil import (
    calculation1, calculation2, calculation3, calculation4
)

# === 1. Загрузка сохранённой модели ===
model = CatBoostRegressor()
model.load_model('catboost_model.cbm')
print("✅ Модель успешно загружена")

# === 2. Загрузка и подготовка данных ===
new_data_path = r'C:\Users\danii\Desktop\итмо\модель\dataset\кампус\output.xlsx'
new_df = pd.read_excel(new_data_path, parse_dates=["Date"])
print(f"Загружено строк: {len(new_df)} | Колонки: {new_df.columns.tolist()}")

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

X_new = new_df_clean[features]

# === 3. Разделение данных на fine-tuning и прогноз ===
if 'target' in new_df_clean.columns:
    y_true = new_df_clean['target'].values
elif 'Electricity load (kW)' in new_df_clean.columns:
    y_true = new_df_clean['Electricity load (kW)'].values
else:
    raise ValueError("Нет столбца с фактическим потреблением ('target' или 'Electricity load (kW)')")

X_ft, X_forecast, y_ft, y_forecast = train_test_split(
    X_new, y_true, test_size=0.75, random_state=42, shuffle=False
)
X_train_ft, X_val_ft, y_train_ft, y_val_ft = train_test_split(
    X_ft, y_ft, test_size=0.2, random_state=42, shuffle=False
)

# === 4. Fine-tuning модели ===
print("\n=== 🛠 Fine-tuning модели на 25% данных ===")
model.fit(
    X_train_ft, y_train_ft,
    eval_set=(X_val_ft, y_val_ft),
    init_model=model,
    early_stopping_rounds=50,
    use_best_model=True,
    verbose=100
)
model.save_model("catboost_model_finetuned.cbm")
print("✅ Fine-tuned модель сохранена")

pred_val = model.predict(X_val_ft)
print(f"\n📊 Метрики на валидации:\nMAE = {mean_absolute_error(y_val_ft, pred_val):.2f}\nRMSE = {np.sqrt(mean_squared_error(y_val_ft, pred_val)):.2f}")

# === 5. Прогноз ===
print("🔮 Выполняется прогноз...")
predictions = model.predict(X_forecast)

results_df = X_forecast.copy()
results_df['time'] = new_df_clean.iloc[X_forecast.index]['Date']
results_df['predicted_load'] = predictions
results_df['fact'] = y_forecast
results_df['year'] = results_df['time'].dt.year
results_df['month'] = results_df['time'].dt.month
results_df.to_excel("new_data_forecast.xlsx", index=False)
print("✅ Прогноз сохранён в: new_data_forecast.xlsx")

# === 6. Загрузка тарифных данных ===
base_path = os.path.dirname(os.path.abspath(__file__))
user_data = pd.read_excel(os.path.join(base_path, "настройки.xlsx"), header=0)
category_contract, v_level, s_level, year_user, company = user_data.iloc[0, [0, 1, 2, 3, 5]]

prices1 = pd.read_excel(os.path.join(base_path, "Почасовые цены1.xlsx"), parse_dates=["time"])
prices2 = pd.read_excel(os.path.join(base_path, "Почасовые цены2.xlsx"), parse_dates=["time"])
hourly_prices = pd.concat([prices1, prices2]).set_index("time").sort_index()

years_in_data = sorted(results_df["year"].unique())
print(f"Создаём копии почасовых цен для годов: {years_in_data}")

def safe_replace_year(dt, new_year):
    try:
        return dt.replace(year=int(new_year))
    except ValueError:
        return dt.replace(year=int(new_year), day=28) if dt.month == 2 and dt.day == 29 else dt

hourly_prices_full = []
for y in years_in_data:
    tmp = hourly_prices.copy()
    tmp.index = tmp.index.map(lambda dt: safe_replace_year(dt, y))
    hourly_prices_full.append(tmp)
hourly_prices = pd.concat(hourly_prices_full).sort_index()

allowed_keys = ['cost_cat1_rub', 'cost_cat22_rub', 'cost_cat23_rub', 'cost_cat3_rub', 'cost_cat4_rub']

# === 7. Основной расчёт ===
all_results = []
cat_dfs = {1: [], 2: [], 3: [], 4: []}

for (year, month) in sorted(results_df[['year', 'month']].drop_duplicates().values.tolist()):
    month_data = results_df[(results_df['year'] == year) & (results_df['month'] == month)]
    res_fact = month_data[['time', 'fact']].rename(columns={'fact': 'consumption'}).set_index('time')
    res_forecast = month_data[['time', 'predicted_load']].rename(columns={'predicted_load': 'consumption'}).set_index('time')

    if res_fact.empty or res_forecast.empty:
        continue

    start, end = res_fact.index.min(), res_fact.index.max()
    mask = (hourly_prices.index >= start) & (hourly_prices.index <= end)
    if hourly_prices.loc[mask].empty:
        continue

    # === Фактические данные ===
    df1_f = calculation1(hourly_prices, res_fact, company, category_contract, v_level, s_level, year_user, year)
    df2_f = calculation2(hourly_prices, res_fact, company, category_contract, v_level, s_level, year_user, year)
    df3_f = calculation3(hourly_prices, res_fact, company, category_contract, v_level, s_level, year_user, year)
    df4_f = calculation4(hourly_prices, res_fact, company, category_contract, v_level, s_level, year_user, year)
    prices_f = pd.concat([df1_f, df2_f, df3_f, df4_f])
    optimal_f = prices_f.loc[allowed_keys].iloc[prices_f.loc[allowed_keys]['value'].argmin()]

    # === Оптимизация по прогнозу, расчёт по факту ===
    df1_tmp = calculation1(hourly_prices, res_forecast, company, category_contract, v_level, s_level, year_user, year)
    df2_tmp = calculation2(hourly_prices, res_forecast, company, category_contract, v_level, s_level, year_user, year)
    df3_tmp = calculation3(hourly_prices, res_forecast, company, category_contract, v_level, s_level, year_user, year)
    df4_tmp = calculation4(hourly_prices, res_forecast, company, category_contract, v_level, s_level, year_user, year)
    prices_tmp = pd.concat([df1_tmp, df2_tmp, df3_tmp, df4_tmp])
    optimal_cat_p = prices_tmp.loc[allowed_keys]['value'].idxmin()

    df_opt_p = {
        'cost_cat1_rub': df1_f,
        'cost_cat22_rub': df2_f,
        'cost_cat23_rub': df2_f,
        'cost_cat3_rub': df3_f,
        'cost_cat4_rub': df4_f
    }[optimal_cat_p]
    optimal_p_cost = df_opt_p.loc[optimal_cat_p, 'value']

    all_results.append({
        "year": year,
        "month": month,
        "fact_optimal_cat": optimal_f.name,
        "fact_optimal_cost": optimal_f.value,
        "forecast_optimal_cat": optimal_cat_p,
        "forecast_optimal_cost": optimal_p_cost,
        "delta_rub": optimal_p_cost - optimal_f.value
    })

    # === Добавляем разницу по каждой категории ===
    for i, df_cat in enumerate([df1_f, df2_f, df3_f, df4_f], start=1):
        df_cat = df_cat.copy()
        df_cat["year"] = year
        df_cat["month"] = month
        df_cat["fact_optimal_cost"] = optimal_f.value
        df_cat["delta_from_fact_optimal"] = df_cat["value"] - optimal_f.value
        cat_dfs[i].append(df_cat)

    del df1_f, df2_f, df3_f, df4_f, df1_tmp, df2_tmp, df3_tmp, df4_tmp
    gc.collect()

# === 8. Сохраняем единый Excel по категориям ===
summary_xlsx_path = os.path.join(base_path, "monthly_fact_dfs_summary.xlsx")
with pd.ExcelWriter(summary_xlsx_path, engine="openpyxl") as writer:
    for i, dfs in cat_dfs.items():
        if dfs:
            pd.concat(dfs, ignore_index=True).to_excel(writer, sheet_name=f"cat{i}", index=False)
print(f"\n📘 Все категории сохранены в: {summary_xlsx_path}")

# === 9. Итоговая таблица ===
df_summary = pd.DataFrame(all_results)
df_yearly = df_summary.groupby("year", as_index=False)["delta_rub"].sum().rename(columns={"delta_rub": "year_total_delta"})
df_summary = df_summary.merge(df_yearly, on="year", how="left")

# Добавляем среднюю разницу по категориям из cat_dfs
for i in range(1, 5):
    df_cat = pd.concat(cat_dfs[i], ignore_index=True)
    diff_mean = (
        df_cat.groupby(["year", "month"])["delta_from_fact_optimal"]
        .mean()
        .reset_index()
        .rename(columns={"delta_from_fact_optimal": f"cat{i}_avg_diff"})
    )
    df_summary = df_summary.merge(diff_mean, on=["year", "month"], how="left")

out_path = os.path.join(base_path, "summary_categories_comparison_by_month.xlsx")
df_summary.to_excel(out_path, index=False)
print(f"📊 Итог сохранён: {out_path}")

# === 10. Совпадения категорий ===
n_total = len(df_summary)
n_match = (df_summary['fact_optimal_cat'] == df_summary['forecast_optimal_cat']).sum()
print(f"\nСовпало категорий: {n_match}/{n_total} ({n_match / n_total * 100:.1f}%)")
print(f"\nВсего месяцев: {n_total}")
print(f"Совпало по категории: {n_match}")
print(f"Процент совпадений: {n_match / n_total * 100:.2f}%")


# === 11. Улучшенные графики по годам ===

years = sorted([y for y in results_df['year'].unique() if 2018 <= y <= 2020])

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


        ax.plot(res_fact.index, res_fact.values, label=" Fact", linewidth=2.5, color='purple')
        ax.plot(res_forecast.index, res_forecast.values, label="Forecast  ", linewidth=2.5, linestyle="--", color='limegreen')

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
