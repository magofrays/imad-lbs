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
print(f"  Фактическое последнее значение: {1.577:.4f}")
print(f"  Тестовый вектор заканчивается в: {data_filled.index[-1]}")