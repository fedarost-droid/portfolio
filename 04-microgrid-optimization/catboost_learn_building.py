import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

# === 1. Загрузка данных ===
file_path = r'C:\Users\danii\Downloads\output.xlsx'
df = pd.read_excel(file_path, parse_dates=["Date"])

# Удалим пропуски (если они есть)
df = df.dropna()
df["Date"] = pd.to_datetime(df["Date"])

# === 2. Разделение на train/val/test ===
train = df[df["Date"] < "2017-01-01"]
val = df[(df["Date"] >= "2017-01-01") & (df["Date"] < "2018-01-01")]
test = df[df["Date"] >= "2018-01-01"]

# === 3. Подготовка признаков и целевой переменной ===
features = [
    "hour", "dayofweek", "month", "dayofyear", "is_weekend",
    "lag_1h", "lag_24h", "lag_168h",
    "rolling_mean_3h", "rolling_mean_24h", "rolling_mean_168h"
]

X_train, y_train = train[features], train["target"]
X_val, y_val = val[features], val["target"]
X_test, y_test = test[features], test["target"]

# === 4. Обучение CatBoost модели ===
model = CatBoostRegressor(
    iterations=1000,
    learning_rate=0.05,
    depth=6,
    eval_metric="MAE",
    early_stopping_rounds=50,
    verbose=100
)

model.fit(
    X_train, y_train,
    eval_set=(X_val, y_val),
    use_best_model=True
)

# === 5. Оценка модели ===
val_pred = model.predict(X_val)
test_pred = model.predict(X_test)

mae_val = mean_absolute_error(y_val, val_pred)
mae_test = mean_absolute_error(y_test, test_pred)

print(f"MAE на валидации: {mae_val:.2f}")
print(f"MAE на тесте: {mae_test:.2f}")

# === 6. Визуализация прогноза по месяцам ===

# Добавляем прогнозы в тестовый датафрейм
test_with_pred = test.copy()
test_with_pred['pred'] = test_pred

# Создаем столбец с годом-месяцем для группировки
test_with_pred['year_month'] = test_with_pred['Date'].dt.to_period('M')

# Получаем уникальные месяцы
unique_months = test_with_pred['year_month'].unique()

# Сортируем месяцы по порядку
unique_months = sorted(unique_months)

# Определяем layout для subplots
n_months = len(unique_months)
n_cols = 2  # 2 графика в строке
n_rows = (n_months + n_cols - 1) // n_cols  # Округление вверх

# Создаем фигуру с subplots
fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
axes = axes.flatten() if n_months > 1 else [axes]  # Преобразуем в плоский массив

for i, month in enumerate(unique_months):
    if i >= len(axes):  # Защита от выхода за границы
        break

    # Фильтруем данные для текущего месяца
    month_data = test_with_pred[test_with_pred['year_month'] == month]

    # Строим график
    axes[i].plot(month_data['Date'], month_data['target'], label='Факт', linewidth=1.5)
    axes[i].plot(month_data['Date'], month_data['pred'], label='Прогноз', linewidth=1.5, linestyle='--')

    axes[i].set_title(f'Прогноз электропотребления: {month}')
    axes[i].set_xlabel('Дата')
    axes[i].set_ylabel('Электропотребление (kW)')
    axes[i].legend()
    axes[i].tick_params(axis='x', rotation=45)
    axes[i].grid(True, alpha=0.3)

# Скрываем пустые subplots если они есть
for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

plt.tight_layout()
plt.show()

# Дополнительно: общий график для всего тестового периода
plt.figure(figsize=(15, 5))
plt.plot(test["Date"], y_test.values, label="Факт", alpha=0.8)
plt.plot(test["Date"], test_pred, label="Прогноз", alpha=0.8)
plt.title("Прогноз электропотребления на тестовом наборе (общий)")
plt.xlabel("Дата")
plt.ylabel("Электропотребление (kW)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# === 8. Сохранение предсказаний в Excel ===
output_df = test_with_pred[['Date', 'pred']].copy()
output_df.rename(columns={'Date': 'time', 'pred': 'load'}, inplace=True)

output_path = "forecast_output.xlsx"
output_df.to_excel(output_path, index=False)

print(f"Прогноз сохранен в файл: {output_path}")
# === 9. Сохранение модели ===
model.save_model('catboost_model.cbm')