import pandas as pd
import os
from pathlib import Path

# Путь к папке с файлами
folder_path = 'C:\\Users\\danii\\Desktop\\итмо\\модель\\dataset\\кампус\\'

# Получаем список всех Excel файлов в папке
excel_files = list(Path(folder_path).glob('*.xlsx')) + list(Path(folder_path).glob('*.xls'))

if not excel_files:
    raise ValueError("В папке не найдено Excel файлов")

# Сортируем файлы для определенности
excel_files.sort()

print(f"Найдено файлов: {len(excel_files)}")
for i, file in enumerate(excel_files):
    print(f"{i + 1}. {file.name}")

# Читаем первый файл с заголовками
first_file = excel_files[0]
print(f"\nЧитаем основной файл с заголовками: {first_file.name}")
df = pd.read_excel(first_file)

# Читаем и объединяем остальные файлы БЕЗ заголовков
for file in excel_files[1:]:
    print(f"Добавляем файл без заголовков: {file.name}")
    try:
        # Читаем без заголовков, используя те же колонки что и в первом файле
        temp_df = pd.read_excel(file, header=None)

        # Проверяем совпадение количества колонок
        if len(temp_df.columns) != len(df.columns):
            print(f"Предупреждение: файл {file.name} имеет {len(temp_df.columns)} колонок вместо {len(df.columns)}")
            # Если колонок меньше - добавляем NaN, если больше - обрезаем
            if len(temp_df.columns) < len(df.columns):
                for i in range(len(temp_df.columns), len(df.columns)):
                    temp_df[i] = None
            else:
                temp_df = temp_df.iloc[:, :len(df.columns)]

        # Устанавливаем правильные названия колонок
        temp_df.columns = df.columns
        df = pd.concat([df, temp_df], ignore_index=True)

    except Exception as e:
        print(f"Ошибка при чтении файла {file.name}: {e}")
        continue

print(f"\nОбщий размер данных после объединения: {len(df)} строк")

# Обработка даты и создание признаков
df["Date"] = pd.to_datetime(df["Date"], format="%Y %m-%d %H:%M")

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
output_path = os.path.join(folder_path, 'output.xlsx')
df_clean.to_excel(output_path, index=False)
print(f"Результат сохранен в: {output_path}")

df_clean.head(10)