import pandas as pd
import os
from pathlib import Path

# Загружаем файл
df = pd.read_excel("C:\\Users\\danii\\Downloads\\consumption.xlsx")


# Преобразуем дату
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

# Сортируем по времени на случай, если что-то не так
df = df.sort_values("Date").reset_index(drop=True)

# Добавим календарные признаки
df["hour"] = df["Date"].dt.hour
df["dayofweek"] = df["Date"].dt.dayofweek
df["month"] = df["Date"].dt.month
df["dayofyear"] = df["Date"].dt.dayofyear
df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)

# Добавим лаги потребления электроэнергии (1ч, 24ч, 168ч)
for lag in [1, 24, 168]:
    df[f"lag_{lag}h"] = df["Electricity load (kW)"].shift(lag)

# Добавим скользящие средние
df["rolling_mean_3h"] = df["Electricity load (kW)"].shift(1).rolling(window=3).mean()
df["rolling_mean_24h"] = df["Electricity load (kW)"].shift(1).rolling(window=24).mean()
df["rolling_mean_168h"] = df["Electricity load (kW)"].shift(1).rolling(window=168).mean()

# Целевая переменная: прогноз на 1 час вперёд
df["target"] = df["Electricity load (kW)"].shift(-1)

# Убираем строки с NaN (из-за лагов и скользящих средних)
df_clean = df.dropna().reset_index(drop=True)

print(f"Размер данных после очистки: {len(df_clean)} строк")

# Сохраняем результат
output_path = os.path.join('output2.xlsx')
df_clean.to_excel(output_path, index=False)
print(f"Результат сохранен в: {output_path}")

df_clean.head(10)