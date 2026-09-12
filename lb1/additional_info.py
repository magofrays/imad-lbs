"""
Домашнее задание: первичный анализ и предобработка природного временного ряда.
Станция: AATB (Alma-Ata B)
Данные: lb1_data.csv (одноминутное разрешение)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# 1. ЗАГРУЗКА ДАННЫХ
# ----------------------------------------------------------------------

FILE_NAME = 'lb1_data.csv'

try:
    # Читаем файл, пропуская строки, начинающиеся с '#'
    df = pd.read_csv(
        FILE_NAME,
        sep=';',
        comment='#',
        skipinitialspace=True,
        engine='python',
        skiprows=1
    )
except Exception as e:
    print(f"Ошибка при чтении файла: {e}")
    print("Проверьте, что файл существует и имеет формат NMDB (start_date_time;RCORR_E).")
    exit(1)

# Переименуем колонки для удобства (если они ещё не переименованы)
df.columns = [c.strip() for c in df.columns]
if 'start_date_time' not in df.columns:
    # Если заголовок другой, попробуем взять первые две колонки
    df = df.iloc[:, :2]
    df.columns = ['start_date_time', 'RCORR_E']

# Преобразуем время в datetime
df['start_date_time'] = pd.to_datetime(df['start_date_time'], errors='coerce')
df = df.dropna(subset=['start_date_time'])  # удаляем строки с некорректным временем
df = df.set_index('start_date_time').sort_index()

# Переименуем значение в 'value' для удобства
df = df.rename(columns={df.columns[0]: 'value'})
data = df['value'].astype(float)

gaps = data.index.to_series().diff()
gaps = gaps[gaps > pd.Timedelta(minutes=1)]

print("\n   Реальные разрывы во времени (пропуски):")
for ts, delta in gaps.items():
    minutes_missing = int(delta.total_seconds() / 60) - 1
    prev_ts = ts - delta
    print(f"   {prev_ts}  →  {ts}   (пропущено {minutes_missing} мин.)")