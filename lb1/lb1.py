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

print("=" * 60)
print("1. ДАННЫЕ ЗАГРУЖЕНЫ")
print(f"   Всего строк с данными: {len(data)}")
print(f"   Период: с {data.index.min()} по {data.index.max()}")
print("=" * 60)

# ----------------------------------------------------------------------
# 2. ОБНАРУЖЕНИЕ ПРОПУСКОВ И ВЫБРОСОВ (3σ)
# ----------------------------------------------------------------------

# --- 2.1. Пропуски ---
full_range = pd.date_range(start=data.index.min(), end=data.index.max(), freq='1min')
missing_times = full_range.difference(data.index)
num_missing = len(missing_times)
total_points = len(full_range)          # общее число минут в периоде
share_missing = num_missing / total_points * 100

print("\n2. ОБНАРУЖЕНИЕ ПРОПУСКОВ И ВЫБРОСОВ")
print("-" * 60)
print(f"   Количество пропусков: {num_missing}")
print(f"   Доля пропусков от всех обрабатываемых данных: {share_missing:.4f}%")

# --- 2.2. Выбросы (3σ) ---
mean_val = data.mean()
std_val = data.std()

lower_bound = mean_val - 3 * std_val
upper_bound = mean_val + 3 * std_val

outliers_mask = (data < lower_bound) | (data > upper_bound)
num_outliers = outliers_mask.sum()
share_outliers = num_outliers / len(data) * 100

print(f"   Среднее значение: {mean_val:.4f}")
print(f"   Стандартное отклонение (σ): {std_val:.4f}")
print(f"   Границы 3σ: [{lower_bound:.4f}; {upper_bound:.4f}]")
print(f"   Количество выбросов (3σ): {num_outliers}")
print(f"   Доля выбросов от всех обрабатываемых данных: {share_outliers:.4f}%")
print("-" * 60)

# ----------------------------------------------------------------------
# 3. ВЫЧИСЛЕНИЕ МЕДИАНЫ ПО ДАННЫМ БЕЗ ПРОПУСКОВ И ВЫБРОСОВ
# ----------------------------------------------------------------------
data_clean = data[~outliers_mask]

median_val = np.median(data_clean)
print(f"\n3. МЕДИАННОЕ ЗНАЧЕНИЕ (без пропусков и выбросов): {median_val:.4f}")

# ----------------------------------------------------------------------
# 4. ЗАМЕНА ПРОПУСКОВ И ВЫБРОСОВ МЕДИАННЫМИ ЗНАЧЕНИЯМИ
# ----------------------------------------------------------------------
# Создаём новый ряд на полной временной сетке
data_filled = data.reindex(full_range)

# Заменяем пропуски (NaN) на медиану
data_filled = data_filled.fillna(median_val)

# Заменяем выбросы на медиану.
# Для этого нужно сопоставить индексы выбросов с полным рядом.
outlier_indices = data.index[outliers_mask]
data_filled.loc[outlier_indices] = median_val

print(f"\n4. ЗАМЕНА ПРОПУСКОВ И ВЫБРОСОВ")
print(f"   Все пропуски ({num_missing} шт.) и выбросы ({num_outliers} шт.) заменены на {median_val:.4f}")
print(f"   Итоговый ряд содержит {len(data_filled)} точек (полная временная сетка).")

# ----------------------------------------------------------------------
# 5. ПОСТРОЕНИЕ ГРАФИКОВ (отдельными картинками)
# ----------------------------------------------------------------------

# --- График 1: исходный ряд ---
plt.figure(figsize=(14, 5))
plt.plot(data.index, data.values, 'b.', markersize=1, label='Исходные данные')
plt.title('Исходный временной ряд (AATB, 02–10 мая 2017)')
plt.xlabel('Время')
plt.ylabel('RCORR_E')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig('original_series.png', dpi=150)
plt.close()   # закрываем фигуру, чтобы не мешала следующей
print("График исходного ряда сохранён: original_series.png")

# --- График 2: очищенный ряд ---
plt.figure(figsize=(14, 5))
plt.plot(data_filled.index, data_filled.values, 'r.', markersize=1, label='Очищенные данные')
plt.title('Очищенный временной ряд (пропуски и выбросы заменены медианой)')
plt.xlabel('Время')
plt.ylabel('RCORR_E')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig('cleaned_series.png', dpi=150)
plt.close()
print("График очищенного ряда сохранён: cleaned_series.png")

# --- (опционально) Совмещённый график для отчёта ---
plt.figure(figsize=(14, 8))

plt.subplot(2, 1, 1)
plt.plot(data.index, data.values, 'b.', markersize=1, label='Исходные данные')
plt.title('Исходный временной ряд (AATB, 02–10 мая 2017)')
plt.ylabel('RCORR_E')
plt.grid(True, alpha=0.3)
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(data_filled.index, data_filled.values, 'r.', markersize=1, label='Очищенные данные')
plt.title('Очищенный временной ряд (пропуски и выбросы заменены медианой)')
plt.xlabel('Время')
plt.ylabel('RCORR_E')
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
plt.savefig('time_series_comparison.png', dpi=150)
plt.close()
print("Совмещённый график сохранён: time_series_comparison.png")

print("\n5. ГРАФИКИ ПОСТРОЕНЫ")

# ----------------------------------------------------------------------
# 6. ВЫВОДЫ
# ----------------------------------------------------------------------
print("\n" + "=" * 60)
print("6. ВЫВОДЫ")
print("=" * 60)
print(f"""
В ходе работы были проанализированы данные нейтронного монитора AATB
за период со 02.05.2017 по 10.05.2017 (одноминутное разрешение).

1. Обнаружено {num_missing} пропусков ({share_missing:.4f}% от всех данных).
2. Обнаружено {num_outliers} выбросов по правилу 3σ ({share_outliers:.4f}%).
3. Медианное значение по "чистым" данным составило {median_val:.4f}.
4. Все пропуски и выбросы заменены на медианное значение.
5. Построены графики исходного и очищенного временных рядов.
   На графике очищенного ряда видны "вставки" на месте пропусков,
   что делает ряд пригодным для дальнейшего анализа.
""")
print("=" * 60)