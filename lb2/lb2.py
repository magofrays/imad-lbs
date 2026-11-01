import pandas as pd
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.spatial.distance import cdist
from sklearn.preprocessing import MinMaxScaler



FILE_NAME = 'lb2_data.csv'
try:
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
df.columns = [c.strip() for c in df.columns]
if 'start_date_time' not in df.columns:
    df = df.iloc[:, :2]
    df.columns = ['start_date_time', 'value']

df['start_date_time'] = pd.to_datetime(df['start_date_time'], errors='coerce')
df = df.dropna(subset=['start_date_time'])  
df = df.set_index('start_date_time').sort_index()

data = df['value'].astype(float)
full_range = pd.date_range(start=data.index.min(), end=data.index.max(), freq='1min')
missing_times = full_range.difference(data.index)
data_filled = data.reindex(full_range)
median_val = data.mean()
data_filled = data_filled.fillna(median_val)


m = 180
values = data_filled.values
all_vectors = sliding_window_view(values, window_shape=m)
test_vector = all_vectors[-1] 
history_vectors = all_vectors[:-1]

k = 3

scaler = MinMaxScaler()
history_scaled = scaler.fit_transform(history_vectors)
test_scaled = scaler.transform(test_vector.reshape(1, -1)).flatten()
euclidean_dists = cdist(history_scaled, test_scaled.reshape(1, -1), metric='euclidean').flatten()
manhattan_dists = cdist(history_scaled, test_scaled.reshape(1, -1), metric='cityblock').flatten()
cosine_dists = cdist(history_scaled, test_scaled.reshape(1, -1), metric='cosine').flatten()

k_nearest_euclidean = np.argsort(euclidean_dists)[:k]
k_nearest_manhattan = np.argsort(manhattan_dists)[:k]
k_nearest_cosine = np.argsort(cosine_dists)[:k]

def predict_next_value(history_indices, m, data_series):
    next_values = []
    for idx in history_indices:
        next_idx = idx + m
        if next_idx < len(data_series):
            next_values.append(data_series.iloc[next_idx])
        else:
            print(f"Предупреждение: для соседа {idx} нет следующего значения")
    
    if len(next_values) == 0:
        return None
    
    return np.mean(next_values)

pred_euclidean = predict_next_value(k_nearest_euclidean, m, data_filled)
pred_manhattan = predict_next_value(k_nearest_manhattan, m, data_filled)
pred_cosine    = predict_next_value(k_nearest_cosine, m, data_filled)

print("=" * 60)
print("ПРОГНОЗ СЛЕДУЮЩЕГО ЗНАЧЕНИЯ РЯДА")
print("=" * 60)
print(f"  Метрика 'Евклид':      {pred_euclidean:.4f}")
print(f"  Метрика 'Манхэттен':   {pred_manhattan:.4f}")
print(f"  Метрика 'Косинус':     {pred_cosine:.4f}")
print("-" * 60)
print(f"  Фактическое последнее значение: {values[-1]:.4f}")
print(f"  Тестовый вектор заканчивается в: {data_filled.index[-1]}")

# --------------------- Построение графиков ---------------------
import matplotlib
matplotlib.use('Agg')
import matplotlib.dates as mdates

import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 11,
    'axes.labelsize': 11,
    'legend.fontsize': 10,
    'axes.grid': True,
    'grid.alpha': 0.35,
})

times = data_filled.index
test_start_time = times[-m]

# Рисунок 1: исходный временной ряд с выделенным тестовым вектором
fig, ax = plt.subplots(figsize=(10, 3.2))
ax.plot(times, values, lw=0.6, color='tab:blue')
ax.axvspan(test_start_time, times[-1], color='tab:orange', alpha=0.35,
           label='тестовый вектор (180 отсчётов)')
ax.set_title('Исходный временной ряд, станция LMKS, 1–14 февраля 2026 г.')
ax.set_xlabel('Дата')
ax.set_ylabel('Значение ряда')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%d.%m'))
ax.legend(loc='upper left')
fig.tight_layout()
fig.savefig('fig1_original_series.png', dpi=200)
plt.close(fig)

metric_names = ['Евклидова', 'Манхэттенская', 'Косинусная']
neighbor_sets = [k_nearest_euclidean, k_nearest_manhattan, k_nearest_cosine]
neighbor_colors = ['tab:red', 'tab:green', 'tab:purple']

# Рисунок 2: тестовый вектор и ближайшие соседи, наложенные друг на друга
fig, axes = plt.subplots(3, 1, figsize=(9, 5.2), sharex=True)
x = np.arange(m)
for ax, name, neigh in zip(axes, metric_names, neighbor_sets):
    ax.plot(x, test_vector, color='black', lw=1.8, label='тестовый вектор')
    for c, idx in zip(neighbor_colors, neigh):
        t0 = times[idx].strftime('%d.%m %H:%M')
        ax.plot(x, history_vectors[idx], color=c, lw=0.9, alpha=0.85,
                label=f'сосед {t0}')
    ax.set_title(f'Метрика: {name.lower()}', fontsize=10)
    ax.set_ylabel('Значение', fontsize=10)
    ax.tick_params(labelsize=9)
    ax.legend(loc='upper right', ncol=2)
axes[-1].set_xlabel('Номер отсчёта внутри вектора')
fig.suptitle('Тестовый вектор и его ближайшие соседи (k = 3)')
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig('fig2_nearest_vectors.png', dpi=200)
plt.close(fig)

# Рисунок 3: распределения расстояний до векторов-историй
all_dists = [euclidean_dists, manhattan_dists, cosine_dists]
fig, axes = plt.subplots(1, 3, figsize=(10, 3))
for ax, name, dists, neigh in zip(axes, metric_names, all_dists, neighbor_sets):
    ax.hist(dists, bins=60, color='tab:blue', alpha=0.75)
    for idx in neigh:
        ax.axvline(dists[idx], color='tab:red', ls='--', lw=1.2)
    ax.set_title(name)
    ax.set_xlabel('Расстояние')
axes[0].set_ylabel('Число векторов')
fig.suptitle('Распределения расстояний до векторов-историй')
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig('fig3_distance_histograms.png', dpi=200)
plt.close(fig)

# Рисунок 4: прогноз следующего значения по трём метрикам
tail = 12 * 60  # последние 12 часов наблюдений
fig, ax = plt.subplots(figsize=(10, 3.4))
ax.plot(times[-tail:], values[-tail:], color='tab:blue', lw=1.0, label='наблюдения')
ax.axvspan(test_start_time, times[-1], color='tab:orange', alpha=0.3,
           label='тестовый вектор')
pred_time = times[-1] + pd.Timedelta(minutes=1)
preds = [pred_euclidean, pred_manhattan, pred_cosine]
markers = ['o', 's', '^']
for val, mk, c, name in zip(preds, markers, neighbor_colors, metric_names):
    ax.plot([pred_time], [val], marker=mk, ms=8, color=c,
            label=f'прогноз ({name.lower()}): {val:.3f}')
ax.plot([times[-1]], [values[-1]], marker='*', ms=11, color='black',
        label=f'последнее наблюдение: {values[-1]:.3f}')
ax.set_title('Прогноз значения ряда на 15.02.2026 00:00 по ближайшим соседям')
ax.set_xlabel('Дата, время')
ax.set_ylabel('Значение ряда')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%d.%m %H:%M'))
ax.legend(loc='upper left')
fig.tight_layout()
fig.savefig('fig4_forecast.png', dpi=200)
plt.close(fig)

print("Графики сохранены: fig1_original_series.png, fig2_nearest_vectors.png, "
      "fig3_distance_histograms.png, fig4_forecast.png")