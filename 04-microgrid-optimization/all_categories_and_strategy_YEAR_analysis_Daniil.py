import pandas as pd
import sqlalchemy
import numpy as np
import regulation_algorithm
from datetime import date, time, timedelta
import datetime
import time
import matplotlib.pyplot as plt
import os

def run_file():

    month_range = range(3,13)

    path = os.path.abspath(os.path.join(os.getcwd(), "Профили Даниил"))
    files = os.listdir(path)  # list files
    files_xls = [f for f in files if (f[-3:] == 'xls') | (f[-4:] == 'xlsx')]  # pick out 'xls' and 'xlsx' files
    all_prof = pd.DataFrame()
    for f in files_xls:
        f = 'umnik.xlsx'
        path = r'./../EnergySolution/Профили Даниил/' + f

        start_time = time.time()
        cons = pd.read_excel(path)
        cons.columns = ['time','consumption']
        # % Prepare data and read settings
        user_data = pd.read_excel(r'./../Tarrifs_new/data/настройки.xlsx', header=0)
        category_contract = user_data.iloc[0, 0]
        v_level = user_data.iloc[0, 1]
        s_level = user_data.iloc[0, 2]
        year = user_data.iloc[0, 3]
        month = user_data.iloc[0, 4]
        cheap1 = 0
        cheap2 = 0
        cheap3 = 0
        cheap4 = 0
        ck_year = pd.DataFrame()
        result_cons = pd.DataFrame()
        price_months = pd.DataFrame()
        res_cons_start = cons.copy()

        if cons.shape[1] == 2:
            res_cons = cons
            if res_cons.time[50].year == 2019:
                res_cons['time'] = pd.to_datetime(res_cons.time) + pd.to_timedelta(365*24, unit='h')
            res_cons.set_index(res_cons.time, inplace=True)
            res_cons.drop(columns='time', inplace=True)

        ind = res_cons[res_cons.index.day == 30].index + pd.to_timedelta(24, unit='h')
        tab = res_cons[res_cons.index.day == 30]
        tab = tab.set_index(ind)
        res_cons = res_cons.append(tab)
        cons_start = res_cons.copy()
        # determine company from the user data for downloading prices
        comps = pd.read_excel(r'./../Tarrifs_new/data/Компании.xlsx', header=0)
        company = comps[comps.Number == user_data.iloc[:, 5][0]].Company[0]

        # чтение цен
        hourly_prices = pd.read_parquet(r'./../Tarrifs_new/prices/' + company + '/' + str(year) + '/Почасовые цены')
        for m in month_range:
            month = m
            res_cons = cons_start[cons_start.index.month == month]
            # %% расчет БЕЗ накопителя по профилю потребления
            df1_all_clear = calculation1(hourly_prices, res_cons,company,category_contract,v_level,s_level,year,month)
            df2_all_clear = calculation2(hourly_prices, res_cons, company, category_contract, v_level, s_level, year, month)
            df3_all_clear = calculation3(hourly_prices, res_cons, company, category_contract, v_level, s_level, year, month)
            df4_all_clear = calculation4(hourly_prices, res_cons, company, category_contract, v_level, s_level, year, month)

            prices0 = pd.concat([df1_all_clear, df2_all_clear, df3_all_clear, df4_all_clear],axis=0)
            prices0 = prices0.loc[['cost_cat1_rub','cost_cat22_rub', 'cost_cat23_rub','cost_cat3_rub','cost_cat4_rub']]
            prices0.index = [1,22,23,3,4]
            price_all = prices0

            # price_all = price_all[price_all.index == 3]

            price_all['key'] = price_all.index
            optimal = price_all.iloc[price_all['value'].argmin()]

            ck_month = pd.DataFrame(columns=['month', 'ck', 'cost'])
            ck_month.loc[month, ['month', 'ck','cost']] = [m, optimal.key, optimal.value]
            ck_month.loc[month, ['month', 'ck','cost']] = [m, optimal.key,optimal.value]

            ck_year = ck_year.append(ck_month)
            price_all.reset_index(drop=True, inplace=True)
            price_all['month'] = m
            price_months = price_months.append(price_all)

        ck_year.set_index('month',inplace=True)
        ck_year.loc['inf'] = [f, ck_year.cost.sum()]
        print(price_months)
        print(f)
        print(ck_year)


def calculation1(hourly_prices, res_cons, company,category_contract, v_level,s_level,year,month):
    month = res_cons.index.month[0]
    # %% 1 category
    price_cat1_rub_mwh = hourly_prices[(hourly_prices.volt == v_level) & (hourly_prices.max_s == s_level) &
                                       (hourly_prices.index.month == month) & (
                                               hourly_prices.type == category_contract) & (
                                               hourly_prices.ck == 1)].value.iloc[0]

    total_cons = res_cons.consumption.sum() / 1000
    cost_cat1_rub = price_cat1_rub_mwh * total_cons

    df_all = pd.DataFrame({'key': ['cost_cat1_rub'],'value': [cost_cat1_rub]}).set_index('key')

    return (df_all)

def calculation2(hourly_prices, res_cons, company, category_contract, v_level, s_level, year, month):
    month = res_cons.index.month[0]
    # %% 2_category
    price_two_two = hourly_prices[(hourly_prices.volt == v_level) & (hourly_prices.max_s == s_level) &
                                  (hourly_prices.index.month == month) & (hourly_prices.type == category_contract) & (
                                          hourly_prices.ck == 2)].sort_index()

    price_two_three = hourly_prices[(hourly_prices.volt == v_level) & (hourly_prices.max_s == s_level) &
                                    (hourly_prices.index.month == month) & (hourly_prices.type == category_contract) & (
                                            hourly_prices.ck == 3)].sort_index()

    cost_two = (price_two_two['value'] * res_cons['consumption']) / 1000

    cost_three = (price_two_three['value'] * res_cons['consumption']) / 1000

    cost_cat3peak_rub = cost_three[((cost_three.index.hour < 10) & (cost_three.index.hour >= 7)) |
                                   ((cost_three.index.hour < 21) & (cost_three.index.hour >= 17))].sum()
    cost_cat3halfpeak_rub = cost_three[((cost_three.index.hour < 17) & (cost_three.index.hour >= 10)) |
                                       ((cost_three.index.hour < 23) & (cost_three.index.hour >= 21))].sum()
    cost_cat3night_rub = cost_three[(cost_three.index.hour == 23) | (cost_three.index.hour < 7)].sum()
    cost_cat23_rub = cost_cat3peak_rub + cost_cat3halfpeak_rub + cost_cat3night_rub

    cost_cat2night_rub = cost_two[(cost_two.index.hour == 23) | (cost_two.index.hour < 7)].sum()
    cost_cat2day_rub = cost_two[(cost_two.index.hour < 23) & (cost_two.index.hour >= 7)].sum()
    cost_cat22_rub = cost_cat2day_rub + cost_cat2night_rub

    df_all = pd.DataFrame({'key': ['cost_cat2day_rub', 'cost_cat2night_rub','cost_cat22_rub',
                                   'cost_cat3halfpeak_rub', 'cost_cat3peak_rub', 'cost_cat3night_rub',
                                   'cost_cat23_rub',],
                           'value': [cost_cat2day_rub, cost_cat2night_rub, cost_cat22_rub,
                                     cost_cat3halfpeak_rub, cost_cat3peak_rub, cost_cat3night_rub,
                                     cost_cat23_rub]}).set_index('key')

    return (df_all)

def calculation3(hourly_prices, res_cons, company, category_contract, v_level, s_level, year, month):
    month = res_cons.index.month[0]
    # %% 3_category
    # 1 component
    price_three = hourly_prices[(hourly_prices.volt == v_level) &
                                (hourly_prices.max_s == s_level) &
                                (hourly_prices.index.month == month) &
                                (hourly_prices.type == category_contract) &
                                (hourly_prices.ck == 4)]

    cost_three = (price_three['value'] * res_cons['consumption']) / 1000
    cost_cat4energy_rub = cost_three.sum()
    # 2 component
    # 2 component
    base_path = os.path.dirname(os.path.abspath(__file__))
    region_file = os.path.join(base_path, "Часы региона.xlsx")
    region_hours = pd.read_excel(region_file)

    # Приводим названия колонок к нижнему регистру (убираем пробелы)
    region_hours.columns = [str(c).strip().lower() for c in region_hours.columns]

    if "time" in region_hours.columns:
        region_hours["time"] = pd.to_datetime(region_hours["time"])
    elif {"day", "hour"}.issubset(region_hours.columns):
        region_hours["time"] = pd.to_datetime(region_hours["day"]) + pd.to_timedelta(region_hours["hour"], unit="h")
    else:
        raise ValueError(f"Файл {region_file} должен содержать либо 'time', либо 'day'+'hour'. "
                         f"Сейчас есть: {region_hours.columns.tolist()}")

    region_hours.set_index("time", inplace=True)
    region_hours = region_hours[region_hours.index.month == month]

    common_index = res_cons.index.intersection(region_hours.index)

    if common_index.empty:
        powergenerator_kw = 0
    else:
        powergenerator_kw = res_cons.loc[common_index, 'consumption'].mean()

    # 🔧 читаем генерацию в отдельный датафрейм
    gen_file = os.path.join(base_path, "Генерация.xlsx")
    gen_prices = pd.read_excel(gen_file, parse_dates=['time'])
    gen_prices.set_index("time", inplace=True)

    price_powergenerator_rub_mw = gen_prices[(gen_prices.max_s == s_level) &
                                             (gen_prices.index.month == month) &
                                             (gen_prices.type == category_contract) &
                                             (gen_prices.ck == 4)].consumption.iloc[0]

    cost_powergeneration_rub = round(powergenerator_kw) * price_powergenerator_rub_mw / 1000
    cost_cat3_rub = cost_powergeneration_rub + cost_cat4energy_rub

    df_all = pd.DataFrame({'key': ['cost_cat3_rub', 'cost_powergeneration_rub'],
                           'value': [cost_cat3_rub, cost_powergeneration_rub]}).set_index('key')

    return df_all


def calculation4(hourly_prices, res_cons, company, category_contract, v_level, s_level, year, month):
    month = res_cons.index.month[0]

    # 1 компонент
    price_four = hourly_prices[(hourly_prices.volt == v_level) &
                               (hourly_prices.max_s == s_level) &
                               (hourly_prices.index.month == month) &
                               (hourly_prices.type == category_contract) &
                               (hourly_prices.ck == 5)]

    cost_four = (price_four['value'] * res_cons['consumption']) / 1000
    cost_cat4energy_rub = cost_four.sum()

    # === Часы региона
    base_path = os.path.dirname(os.path.abspath(__file__))
    region_file = os.path.join(base_path, "Часы региона.xlsx")
    region_hours = pd.read_excel(region_file)
    region_hours.columns = [str(c).strip().lower() for c in region_hours.columns]

    if {"day", "hour"}.issubset(region_hours.columns):
        region_hours["time"] = pd.to_datetime(region_hours["day"]) + pd.to_timedelta(region_hours["hour"], unit="h")
    elif "time" in region_hours.columns:
        region_hours["time"] = pd.to_datetime(region_hours["time"])
    else:
        raise ValueError(f"Файл {region_file} должен содержать либо 'time', либо 'day+hour'. "
                         f"Сейчас есть: {region_hours.columns.tolist()}")

    region_hours.set_index("time", inplace=True)
    region_hours = region_hours[region_hours.index.month == month]

    common_index = res_cons.index.intersection(region_hours.index)
    if common_index.empty:
        print(f"⚠️ Нет совпадений по 'Часы региона' в месяце {month}")
        powergenerator_kw = 0
    else:
        powergenerator_kw = res_cons.loc[common_index, 'consumption'].mean()

    # === Генерация
    gen_file = os.path.join(base_path, "Генерация.xlsx")
    gen_prices = pd.read_excel(gen_file, parse_dates=['time'])
    gen_prices.set_index("time", inplace=True)

    price_powergenerator_rub_mw = gen_prices[(gen_prices.max_s == s_level) &
                                             (gen_prices.index.month == month) &
                                             (gen_prices.type == category_contract) &
                                             (gen_prices.ck == 4)].consumption.iloc[0]

    cost_powergeneration_rub = round(powergenerator_kw) * price_powergenerator_rub_mw / 1000

    # === Транспорт
    holydays = os.path.join(base_path, "нерабочие дни.xlsx")
    hd = pd.read_excel(holydays)
    hd.columns = ['hols']
    hd.set_index('hols', inplace=True)

    transport = pd.read_excel(os.path.join(base_path, "Транспорт.xlsx"))
    if "time" in transport.columns:
        transport["time"] = pd.to_datetime(transport["time"])
        transport.set_index("time", inplace=True)
    else:
        # если 'time' нет → пробуем использовать текущий индекс как даты
        try:
            transport.index = pd.to_datetime(transport.index)
        except Exception:
            raise ValueError("Файл 'Транспорт' должен содержать колонку 'time' или индекс с датами")
    price_powertranspower_rub_mw = transport[(transport.max_s == s_level) &
                                             (transport.index.month == month) &
                                             (transport.type == category_contract) &
                                             (transport.ck == 5) &
                                             (transport.volt == v_level)].consumption.iloc[0]

    load_hrs = pd.read_excel(
        os.path.join(base_path, "часы пиковой нагрузки.xlsx"),
        usecols=[month - 1]).dropna()

    cons1 = res_cons.copy()
    cons1['day'] = pd.to_datetime(res_cons.index.strftime('%Y-%m-%d'))
    cons1['hour'] = res_cons.index.hour + 1

    cons_T = pd.pivot_table(cons1, values='consumption', index='day', columns='hour')
    cons_T = cons_T.loc[cons_T.index.difference(hd.index), :]
    powertranspower_kw = round(
        (cons_T.loc[cons_T.index.dayofweek < 5, load_hrs.iloc[:, 0].values]).max(axis=1).mean()
    )

    cost_powertranspower_rub = powertranspower_kw * price_powertranspower_rub_mw / 1000
    cost_cat4_rub = cost_powertranspower_rub + cost_powergeneration_rub + cost_cat4energy_rub

    df_all = pd.DataFrame({
        'key': ['cost_cat4_rub','cost_cat4energy_rub',
                'cost_powergeneration_rub','cost_powertranspower_rub'],
        'value': [cost_cat4_rub, cost_cat4energy_rub,
                  cost_powergeneration_rub, cost_powertranspower_rub]
    }).set_index('key')

    return df_all



