import pandas as pd
import os
import matplotlib.pyplot as plt
from catboost import CatBoostRegressor
from all_categories_and_strategy_YEAR_analysis_Daniil import (
    calculation1, calculation2, calculation3, calculation4
)

def calculate_category():
    base_path = os.path.dirname(os.path.abspath(__file__))

    # === Загружаем обработанный датасет ===
    df = pd.read_excel(os.path.join(base_path, "output2.xlsx"), parse_dates=["Date"])

    features = [
        "hour", "dayofweek", "month", "dayofyear", "is_weekend",
        "lag_1h", "lag_24h", "lag_168h",
        "rolling_mean_3h", "rolling_mean_24h", "rolling_mean_168h"
    ]

    # === Загружаем сохранённую модель ===
    model = CatBoostRegressor()
    model.load_model(os.path.join(base_path, "catboost_model.cbm"))

    # === Делаем прогноз ===
    df["forecast"] = model.predict(df[features])
    df["fact"] = df["target"]
    df = df.set_index("Date")

    # === Настройки ===
    user_data = pd.read_excel(os.path.join(base_path, "настройки.xlsx"), header=0)
    category_contract = user_data.iloc[0,0]
    v_level = user_data.iloc[0,1]
    s_level = user_data.iloc[0,2]
    year = user_data.iloc[0,3]
    company = user_data.iloc[0,5]

    # === Почасовые цены ===
    prices1 = pd.read_excel(os.path.join(base_path,"Почасовые цены1.xlsx"), parse_dates=['time'])
    prices2 = pd.read_excel(os.path.join(base_path,"Почасовые цены2.xlsx"), parse_dates=['time'])
    hourly_prices = pd.concat([prices1, prices2]).set_index('time').sort_index()

    allowed_keys = ['cost_cat1_rub','cost_cat22_rub','cost_cat23_rub','cost_cat3_rub','cost_cat4_rub']
    all_results = []

    # --- создаём общее полотно для 12 графиков ---
    fig, axes = plt.subplots(3, 4, figsize=(20, 12))
    axes = axes.flatten()

    # --- расчет по месяцам ---
    for month in range(1, 13):
        res_fact = df[df.index.month == month][["fact"]].rename(columns={"fact": "consumption"})
        res_forecast = df[df.index.month == month][["forecast"]].rename(columns={"forecast": "consumption"})
        if res_fact.empty or res_forecast.empty:
            axes[month-1].set_visible(False)
            continue

        # === Факт ===
        df1_f = calculation1(hourly_prices,res_fact,company,category_contract,v_level,s_level,year,month)
        df2_f = calculation2(hourly_prices,res_fact,company,category_contract,v_level,s_level,year,month)
        df3_f = calculation3(hourly_prices,res_fact,company,category_contract,v_level,s_level,year,month)
        df4_f = calculation4(hourly_prices,res_fact,company,category_contract,v_level,s_level,year,month)
        prices_f = pd.concat([df1_f,df2_f,df3_f,df4_f])
        optimal_f = prices_f.loc[allowed_keys].iloc[prices_f.loc[allowed_keys]['value'].argmin()]

        # === Прогноз ===
        df1_p = calculation1(hourly_prices,res_forecast,company,category_contract,v_level,s_level,year,month)
        df2_p = calculation2(hourly_prices,res_forecast,company,category_contract,v_level,s_level,year,month)
        df3_p = calculation3(hourly_prices,res_forecast,company,category_contract,v_level,s_level,year,month)
        df4_p = calculation4(hourly_prices,res_forecast,company,category_contract,v_level,s_level,year,month)
        prices_p = pd.concat([df1_p,df2_p,df3_p,df4_p])
        optimal_p = prices_p.loc[allowed_keys].iloc[prices_p.loc[allowed_keys]['value'].argmin()]

        # === график сравнения ===
        ax = axes[month-1]
        ax.plot(res_fact.index, res_fact["consumption"], label="Факт")
        ax.plot(res_forecast.index, res_forecast["consumption"], label="Прогноз", alpha=0.7)
        ax.set_title(f"Месяц {month}")
        ax.tick_params(axis='x', rotation=30)
        ax.grid(True, alpha=0.3)

        if month == 1:  # легенду показываем только на первом графике
            ax.legend()

        row = {
            "month": month,
            "fact_optimal_cat": optimal_f.name,
            "fact_optimal_cost": optimal_f.value,
            "forecast_optimal_cat": optimal_p.name,
            "forecast_optimal_cost": optimal_p.value,
            "delta_rub": optimal_p.value - optimal_f.value
        }
        all_results.append(row)

    plt.tight_layout()
    plt.show()

    df_summary = pd.DataFrame(all_results)
    print("\n=== Сравнение по категориям (факт vs прогноз) ===")
    print(df_summary)

    # --- сохраняем результат в Excel ---
    out_path = os.path.join(base_path, "summary_categories_comparison.xlsx")
    df_summary.to_excel(out_path, index=False)
    print(f"\nСводная таблица сохранена в файл: {out_path}")

    return df_summary

if __name__ == "__main__":
    results = calculate_category()
