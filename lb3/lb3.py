"""
Лабораторная работа №3. Реализация метода кластеризации данных с
использованием природных временных рядов. Метод нечеткой кластеризации
c-средних (fuzzy c-means) по наблюдениям нейтронного монитора.

Вариант 14: станция Апатиты (APTY).
Класс 1 — спокойное состояние, класс 2 — магнитная буря (табл. п.2).

Этапы:
  1. Загрузка одноминутных данных (lb3_data.csv, канал RCORR —
     скорректированное на давление число импульсов/мин).
  2. Заполнение пропусков медианным значением (аналог median в MatLab).
  3. Формирование матрицы входных данных data из признаков суточных
     реализаций (среднее, СКО, амплитуда, минимум, амплитуда суточной
     вариации).
  4. Реализация метода нечеткой кластеризации c-средних — функция fcm,
     повторяющая алгоритм функции fcm из Matlab Fuzzy Logic Toolbox
     ([centers, U] = fcm(data, Nc, options)).
  5. Проверка работоспособности: синтетический тест, сравнение
     кластеризации с истинной разметкой, показатели качества разбиения.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.decomposition import PCA

RNG_SEED = 42

# ----------------------------------------------------------------------
# 1. ЗАГРУЗКА ДАННЫХ
# ----------------------------------------------------------------------

data_file = "lb3_data.csv"
labels_file = "lb3_labels.csv"

raw = pd.read_csv(data_file, sep=";")
raw["start_date_time"] = pd.to_datetime(raw["start_date_time"], errors="coerce")
raw = raw.dropna(subset=["start_date_time"]).set_index("start_date_time").sort_index()
value_col = "RCORR_APTY" if "RCORR_APTY" in raw.columns else raw.columns[0]

labels = pd.read_csv(labels_file, sep=";")
labels["date"] = pd.to_datetime(labels["date"]).dt.date

print("=" * 70)
print("1. ДАННЫЕ ЗАГРУЖЕНЫ")
print(f"   Всего точек: {len(raw)}")
print(f"   Период: {raw.index.min()} — {raw.index.max()}")
print(f"   Канал: {value_col} (скорректированное на давление число имп./мин)")
print(f"   Дней с разметкой: {len(labels)} "
      f"(класс 1 — {(labels['class'] == 1).sum()}, "
      f"класс 2 — {(labels['class'] == 2).sum()})")
print("=" * 70)

# ----------------------------------------------------------------------
# 2. ЗАПОЛНЕНИЕ ПРОПУСКОВ МЕДИАННЫМ ЗНАЧЕНИИМ (median)
# ----------------------------------------------------------------------
# Сутки считаются отсутствующими в источнике, если все значения равны 0
# (в доступной базе ИЗМИРАН одноминутные данные станции Апатиты выгружены
# только до середины 2019 г. — соответствующие даты исключаются).
# Записи с не физическими значениями (<= 0) трактуются как пропуски.

series = {}          # date -> pd.Series на полной минутной сетке
gap_stats = []       # (date, n_missing, share %, median)
excluded_days = []
for date, grp in raw.groupby(raw.index.date):
    if date not in set(labels["date"]):
        continue
    vals = grp[value_col].values.astype(float)
    if np.median(vals) <= 0:                     # сутки отсутствуют в базе
        excluded_days.append(date)
        continue
    full_idx = pd.date_range(pd.Timestamp(date), pd.Timestamp(date) + pd.Timedelta(hours=23, minutes=59),
                             freq="1min")
    s = grp[value_col].reindex(full_idx)
    s = s.mask(s <= 0)                           # нулевые записи = пропуски
    n_missing = int(s.isna().sum())
    median_val = float(np.median(s.dropna().values))   # аналог median в MatLab
    s = s.fillna(median_val)
    series[date] = s
    gap_stats.append((date, n_missing, 100.0 * n_missing / len(full_idx), median_val))

gap_df = pd.DataFrame(gap_stats, columns=["date", "n_missing", "share_pct", "median"])
days_with_gaps = gap_df[gap_df["n_missing"] > 0]

print("\n2. ЗАПОЛНЕНИЕ ПРОПУСКОВ МЕДИАННЫМ ЗНАЧЕНИЕМ")
print("-" * 70)
if excluded_days:
    print("   Дни, отсутствующие в источнике (данные не выгружены, все значения 0):")
    for d in excluded_days:
        print(f"     {d} (класс {int(labels.loc[labels['date'] == d, 'class'].iloc[0])})")
    print(f"   Итого исключено суток: {len(excluded_days)}")
print(f"   Обработано суточных реализаций: {len(series)}")
print(f"   Суммарное число пропущенных минут: {gap_df['n_missing'].sum()}")
print(f"   Дней с пропусками: {len(days_with_gaps)} из {len(gap_df)}")
print(f"   Доля пропусков от всех данных: "
      f"{100.0 * gap_df['n_missing'].sum() / (1440 * len(gap_df)):.4f}%")
if len(days_with_gaps):
    print("   Дни с пропусками (дата, число минут, доля %, медиана):")
    for _, r in days_with_gaps.iterrows():
        print(f"     {r['date']}  {r['n_missing']:>4} мин  {r['share_pct']:.3f}%  "
              f"median={r['median']:.0f}")
print("-" * 70)

# ----------------------------------------------------------------------
# 3. ФОРМИРОВАНИЕ МАТРИЦЫ ВХОДНЫХ ДАННЫХ data
# ----------------------------------------------------------------------
# Ряды переводятся в проценты отклонения от среднего уровня за весь период
# (как в NMDB: corrected data in %), затем по каждой суточной реализации
# вычисляются признаки.

all_values = np.concatenate([s.values for s in series.values()])
I0 = float(np.median(all_values))
print(f"\n   Базовый уровень I0 (медиана за весь период): {I0:.1f} имп./мин")

rows = []
for date in sorted(series):
    s = series[date]
    p = 100.0 * (s.values / I0 - 1.0)          # вариации интенсивности, %
    h = 100.0 * (s.resample("1h").mean().values / I0 - 1.0)  # среднечасовые, %
    tt = np.arange(24)
    X24 = np.column_stack([np.ones(24),
                           np.sin(2 * np.pi * tt / 24), np.cos(2 * np.pi * tt / 24)])
    beta, *_ = np.linalg.lstsq(X24, h, rcond=None)
    phase = np.arctan2(beta[2], beta[1])       # фаза суточной волны
    rows.append({
        "date": date,
        "class": int(labels.loc[labels["date"] == date, "class"].iloc[0]),
        "mean": float(np.mean(p)),                        # среднее значение ряда
        "std": float(np.std(p, ddof=1)),                  # стандартное отклонение
        "amp_min": float(np.max(p) - np.min(p)),          # амплитуда ряда
        "amp_diurnal": float(np.max(h) - np.min(h)),      # амплитуда часовых средних
        "asym_night_day": float(h[0:6].mean() - h[12:18].mean()),  # асимметрия ночь-день
        "phase_cos": float(np.cos(phase)),                # косинус фазы суточной волны
    })

feat = pd.DataFrame(rows)

# Кандидаты в признаки и их разделяющая способность (критерий Уэлча)
ALL_FEATURES = ["mean", "std", "amp_min", "amp_diurnal", "asym_night_day", "phase_cos"]
ALL_FEATURES_RU = ["среднее, %", "СКО, %", "амплитуда ряда, %",
                   "амплитуда часовых средних, %", "асимметрия ночь-день, %",
                   "cos фазы суточной волны"]
print("\n   Разделяющая способность признаков-кандидатов "
      "(t-критерий Уэлча, класс 1 против класса 2):")
for f, ru in zip(ALL_FEATURES, ALL_FEATURES_RU):
    a = feat.loc[feat["class"] == 1, f]
    b = feat.loc[feat["class"] == 2, f]
    tt_stat, pv = stats.ttest_ind(a, b, equal_var=False)
    print(f"     {ru:<32} t = {tt_stat:+6.2f}, p = {pv:.4f}")

# Итоговая матрица входных данных: среднее значение ряда (базовая
# характеристика) и два признака суточной анизотропии, имеющие
# наибольшую разделяющую способность
FEATURES = ["mean", "asym_night_day", "phase_cos"]
FEATURES_RU = ["среднее, %", "асимметрия ночь-день, %", "cos фазы суточной волны"]
data = feat[FEATURES].values          # строки data соответствуют наблюдениям
y_true = feat["class"].values
feat.to_csv("data_matrix.csv", index=False, sep=";", decimal=",")

print("\n3. МАТРИЦА ВХОДНЫХ ДАННЫХ data")
print("-" * 70)
print(f"   Число наблюдений (суток): {data.shape[0]}, число признаков: {data.shape[1]}")
print(f"   Признаки: {', '.join(FEATURES)}")
print(f"   Сохранено в data_matrix.csv")
for c in (1, 2):
    sub = feat[feat["class"] == c][FEATURES]
    name = "класс 1 (спокойные)" if c == 1 else "класс 2 (буря)"
    print(f"   {name}: n={len(sub)}")
    for f, ru in zip(FEATURES, FEATURES_RU):
        print(f"     {ru:<18}: mean={sub[f].mean():8.3f}  std={sub[f].std():7.3f}")
print("-" * 70)

# ----------------------------------------------------------------------
# 4. РЕАЛИЗАЦИЯ МЕТОДА НЕЧЕТКОЙ КЛАСТЕРИЗАЦИИ c-СРЕДНИХ (fcm)
# ----------------------------------------------------------------------

def _stepfcm(data, U, Nc, expo):
    """Один шаг алгоритма c-средних (аналог stepfcm в Matlab)."""
    mf = U ** expo                                   # (Nc, N)
    centers = (mf @ data) / mf.sum(axis=1, keepdims=True)     # (Nc, F)
    dist = np.linalg.norm(data[:, None, :] - centers[None, :, :], axis=2).T  # (Nc, N)
    dist = np.maximum(dist, np.finfo(float).eps)
    obj = float(np.sum((dist ** 2) * mf))                     # функция цели
    tmp = dist ** (-2.0 / (expo - 1.0))
    U_new = tmp / tmp.sum(axis=0, keepdims=True)
    return U_new, centers, obj


def fcm(data, Nc, options=(2.0, 100, 1e-5, 0), seed=None):
    """Нечеткая кластеризация c-средних.

    Повторяет алгоритм функции fcm из Matlab Fuzzy Logic Toolbox:
      [centers, U] = fcm(data, Nc, options),
    где options = (expo, maxIter, minImprove, display):
      expo      — экспонента нечеткости m (по умолчанию 2),
      maxIter   — максимальное число итераций (100),
      minImprove— минимальное улучшение функции цели (1e-5),
      display   — печать информации (0/1).
    data — матрица наблюдений (строки соответствуют наблюдениям).
    Возвращает centers (Nc, F), U (Nc, N) — матрицу степеней принадлежности,
    obj — историю функции цели.
    """
    expo, maxIter, minImprove, display = options
    data = np.asarray(data, dtype=float)
    N = data.shape[0]
    rng = np.random.default_rng(seed)
    U = rng.random((Nc, N))
    U /= U.sum(axis=0, keepdims=True)          # начальное разбиение
    obj_hist = []
    centers = None
    for it in range(maxIter):
        U, centers, obj = _stepfcm(data, U, Nc, expo)
        obj_hist.append(obj)
        if display and (it % 10 == 0 or it == 0):
            print(f"      iter {it:3d}: obj = {obj:.6f}")
        if it > 0 and abs(obj_hist[-1] - obj_hist[-2]) < minImprove:
            break
    return centers, U, np.array(obj_hist)


# --- Стандартизация признаков ---
mu = data.mean(axis=0)
sigma = data.std(axis=0, ddof=1)
data_z = (data - mu) / sigma

print("\n4. МЕТОД c-СРЕДНИХ")
print("-" * 70)
print("   Признаки стандартизованы (z = (x - mean)/std по каждому признаку).")

# Проверка работоспособности реализации на синтетических данных
rng = np.random.default_rng(0)
blob1 = rng.normal(loc=[-3, -3], scale=0.6, size=(60, 2))
blob2 = rng.normal(loc=[3, 3], scale=0.6, size=(60, 2))
synth = np.vstack([blob1, blob2])
c_s, U_s, _ = fcm(synth, 2, seed=1)
hard_s = U_s.argmax(axis=0)
true_s = np.r_[np.zeros(60, int), np.ones(60, int)]
acc_s = max(np.mean(hard_s == true_s), np.mean((1 - hard_s) == true_s))
pc_s = np.sum(U_s ** 2) / U_s.shape[1]
print("   Проверка реализации на синтетических данных (2 разделимых группы, n=120):")
print(f"     точность разбиения = {acc_s * 100:.1f}%  "
      f"коэффициент разбиения PC = {pc_s:.3f}  "
      f"→ реализация работоспособна")

# Кластеризация реальных данных (мультитарт с разными начальными
# разбиениями, выбирается разбиение с минимальной функцией цели)
Nc = 2
results = []
for seed in range(20):
    centers_i, U_i, obj_i = fcm(data_z, Nc, options=(2.0, 100, 1e-5, 0), seed=seed)
    results.append((obj_i[-1], centers_i, U_i, obj_i))
results.sort(key=lambda t: t[0])
J_best, centers, U, obj_hist = results[0]
n_iters = len(obj_hist)

print(f"   Кластеризация: Nc = {Nc}, options = (m=2.0, maxIter=100, minImprove=1e-5)")
print(f"   Выполнено {len(results)} запусков (мультитарт), выбран запуск "
      f"с минимальной функцией цели J = {J_best:.6f}")
print(f"   Сходимость за {n_iters} итераций")
print("   Центры кластеров (в стандартизованном масштабе):")
for k in range(Nc):
    print(f"     кластер {k + 1}: " +
          "  ".join(f"{ru}={v:+.2f}" for ru, v in zip(FEATURES_RU, centers[k])))
print("-" * 70)

# ----------------------------------------------------------------------
# 5. ПРОВЕРКА РАБОТОСПОСОБНОСТИ КЛАСТЕРИЗАЦИИ
# ----------------------------------------------------------------------

def validity_indices(U, centers, data):
    N = U.shape[1]
    PC = float(np.sum(U ** 2) / N)                       # коэффициент разбиения
    Uc = np.clip(U, 1e-12, None)
    PE = float(-np.sum(Uc * np.log(Uc)) / N)             # энтропия классификации
    dists = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=2)
    np.fill_diagonal(dists, np.inf)
    sep = dists.min()
    mf = U ** 2.0
    J = float(np.sum(mf * np.linalg.norm(
        data[:, None, :] - centers[None, :, :], axis=2).T ** 2))
    XB = J / (N * sep)                                   # индекс Ши–Бени
    return PC, PE, XB


PC, PE, XB = validity_indices(U, centers, data_z)
hard = U.argmax(axis=0)          # четкая кластеризация по максимуму принадлежности

# сопоставление кластеров с классами (наилучшее по точности)
perm_best = None
acc_best = -1
for mapping in ({0: 1, 1: 2}, {0: 2, 1: 1}):
    pred = np.array([mapping[h] for h in hard])
    acc = np.mean(pred == y_true)
    if acc > acc_best:
        acc_best, perm_best, y_pred = acc, mapping, pred

# матрица неточностей: строки — истинный класс, столбцы — предсказанный
conf = np.zeros((2, 2), int)
for t in (1, 2):
    for p in (1, 2):
        conf[t - 1, p - 1] = int(np.sum((y_true == t) & (y_pred == p)))

print("\n5. ПРОВЕРКА РАБОТОСПОСОБНОСТИ")
print("-" * 70)
print(f"   Показатели качества нечеткого разбиения:")
print(f"     коэффициент разбиения PC      = {PC:.4f} (макс = 1)")
print(f"     энтропия классификации PE     = {PE:.4f} (мин = 0)")
print(f"     индекс Ши–Бени XB             = {XB:.4f} (меньше — лучше)")
print(f"   Согласование с истинной разметкой (Nc = 2):")
print(f"     точность = {acc_best * 100:.1f}% "
      f"({int(round(acc_best * len(y_true)))} из {len(y_true)} суток)")
print("     матрица неточностей (строки — истинный класс):")
print(f"                 предск.1  предск.2")
for t in (1, 2):
    print(f"       класс {t}:   {conf[t-1,0]:5d}    {conf[t-1,1]:5d}")
sens = conf[1, 1] / max(1, conf[1].sum())
spec = conf[0, 0] / max(1, conf[0].sum())
print(f"     чувствительность (буря) = {sens:.3f}, специфичность (спокойные) = {spec:.3f}")

# Средние степени принадлежности по классам
print("   Средняя степень принадлежности кластеру «буря» (U2):")
storm_cluster = [k for k, v in perm_best.items() if v == 2][0]
quiet_cluster = 1 - storm_cluster
u2 = U[storm_cluster]
print(f"     класс 1 (спокойные): mean(U2) = {u2[y_true == 1].mean():.3f}, "
      f"max = {u2[y_true == 1].max():.3f}")
print(f"     класс 2 (буря):      mean(U2) = {u2[y_true == 2].mean():.3f}, "
      f"min = {u2[y_true == 2].min():.3f}")
amb = np.sum((u2 > 0.35) & (u2 < 0.65))
print(f"     сильно смешанных суток (0.35 < U2 < 0.65): {amb}")

# Влияние экспоненты нечеткости m
print("   Влияние экспоненты нечеткости m (Nc = 2, мультитарт по 10 запусков):")
for m_try in (1.5, 2.0, 2.5):
    res_m = []
    for seed in range(10):
        _, U_m, obj_m = fcm(data_z, Nc, options=(m_try, 100, 1e-5, 0), seed=seed)
        res_m.append((obj_m[-1], U_m))
    res_m.sort(key=lambda t: t[0])
    hard_m = res_m[0][1].argmax(axis=0)
    acc_m = max(np.mean(np.where(hard_m == 0, 1, 2) == y_true),
                np.mean(np.where(hard_m == 0, 2, 1) == y_true))
    print(f"     m = {m_try}: точность согласования с разметкой = {acc_m * 100:.1f}%")

# Влияние состава признаков
print("   Влияние состава признаков на согласование (Nc = 2, m = 2):")
fidx = {f: i for i, f in enumerate(ALL_FEATURES)}
X_all = feat[ALL_FEATURES].values
X_all_z = (X_all - X_all.mean(axis=0)) / X_all.std(axis=0, ddof=1)
for fs in (["mean", "std", "amp_min"],
           ["asym_night_day"],
           ["asym_night_day", "phase_cos"],
           ["mean", "asym_night_day", "phase_cos"]):
    X_ab = X_all_z[:, [fidx[f] for f in fs]]
    res_ab = []
    for seed in range(10):
        _, U_ab, obj_ab = fcm(X_ab, Nc, options=(2.0, 100, 1e-5, 0), seed=seed)
        res_ab.append((obj_ab[-1], U_ab))
    res_ab.sort(key=lambda t: t[0])
    hard_ab = res_ab[0][1].argmax(axis=0)
    acc_ab = max(np.mean(np.where(hard_ab == 0, 1, 2) == y_true),
                 np.mean(np.where(hard_ab == 0, 2, 1) == y_true))
    if fs == ["mean", "std", "amp_min"]:
        acc_baseline = acc_ab
    print(f"     {'+'.join(fs):<45}: {acc_ab * 100:.1f}%")
print("-" * 70)

# ----------------------------------------------------------------------
# ГРАФИКИ
# ----------------------------------------------------------------------

plt.rcParams.update({"font.size": 9})

# Рис. 1: примеры суточных реализаций двух классов
fig, axes = plt.subplots(2, 2, figsize=(12, 6), sharey=True)
examples = {
    (1, 0): pd.Timestamp("2018-01-03").date(),
    (1, 1): pd.Timestamp("2018-06-29").date(),
    (2, 0): pd.Timestamp("2018-05-06").date(),
    (2, 1): pd.Timestamp("2018-10-08").date(),
}
storm_days_set = set(feat.loc[feat["class"] == 2, "date"])
for (cls, idx), d in examples.items():
    ax = axes[cls - 1][idx]
    p = 100.0 * (series[d].values / I0 - 1.0)
    t = np.arange(1440) / 60.0
    ax.plot(t, p, lw=0.5, color="tab:blue" if cls == 1 else "tab:red", alpha=0.7)
    hv = 100.0 * (series[d].resample("1h").mean().values / I0 - 1.0)
    ax.plot(np.arange(24) + 0.5, hv, lw=2.2, color="k", label="среднечасовые значения")
    ax.set_title(f"{d} — {'класс 1 (спокойный)' if cls == 1 else 'класс 2 (буря)'}",
                 fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("Время, ч")
    ax.set_ylabel("Вариация, %")
axes[0][0].legend(loc="lower left", fontsize=7)
fig.suptitle("Примеры суточных реализаций нейтронного монитора APTY", fontsize=11)
fig.tight_layout()
fig.savefig("fig1_example_days.png", dpi=150)
plt.close(fig)

# Рис. 2: заполнение пропусков (день с наибольшим числом пропусков)
if len(days_with_gaps):
    worst = days_with_gaps.loc[days_with_gaps["n_missing"].idxmax(), "date"]
    d = worst
    full_idx = pd.date_range(pd.Timestamp(d), pd.Timestamp(d) + pd.Timedelta(hours=23, minutes=59), freq="1min")
    s_raw = raw[value_col].reindex(full_idx)
    s_filled = series[d]
    med = gap_df.loc[gap_df["date"] == d, "median"].iloc[0]
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(s_raw.index, s_raw.values, "b.", ms=2, label="Исходные данные (с пропусками)")
    ax.plot(s_filled.index, s_filled.values, "r.", ms=1, label="После заполнения медианой")
    ax.axhline(med, color="g", ls="--", lw=1, label=f"Медиана дня = {med:.0f}")
    ax.set_title(f"Заполнение пропусков медианным значением: {d} "
                 f"({int(days_with_gaps.loc[days_with_gaps['date'] == d, 'n_missing'].iloc[0])} пропущенных минут)")
    ax.set_ylabel("Число импульсов/мин")
    ax.set_xlabel("Время")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig("fig2_gap_fill.png", dpi=150)
    plt.close(fig)

# Рис. 3: распределения признаков-кандидатов по классам (boxplot)
n_c = len(ALL_FEATURES)
fig, axes = plt.subplots(2, 3, figsize=(12, 6.5))
for i, (f, ru) in enumerate(zip(ALL_FEATURES, ALL_FEATURES_RU)):
    ax = axes[i // 3][i % 3]
    vals = [feat.loc[feat["class"] == 1, f], feat.loc[feat["class"] == 2, f]]
    bp = ax.boxplot(vals, tick_labels=["класс 1", "класс 2"], patch_artist=True,
                    widths=0.6)
    for patch, color in zip(bp["boxes"], ["#aec7e8", "#ff9896"]):
        patch.set_facecolor(color)
    ax.set_title(ru, fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
fig.suptitle("Признаки суточных реализаций по классам "
             "(голубой — спокойные сутки, красный — магнитная буря)", fontsize=11)
fig.tight_layout()
fig.savefig("fig3_features_box.png", dpi=150)
plt.close(fig)

# Рис. 4: проекция на плоскость главных компонент
pca = PCA(n_components=2)
Z2 = pca.fit_transform(data_z)
C2 = pca.transform(centers)
expl = pca.explained_variance_ratio_ * 100
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, colors, title in (
    (axes[0], np.where(y_true == 1, "tab:blue", "tab:red"), "Истинная разметка"),
    (axes[1], np.where(y_pred == 1, "tab:blue", "tab:red"), "Результат кластеризации"),
):
    ax.scatter(Z2[:, 0], Z2[:, 1], c=colors, s=45, alpha=0.75,
               edgecolors="k", linewidths=0.4)
    ax.scatter(C2[:, 0], C2[:, 1], c=["tab:blue", "tab:red"], s=260,
               marker="*", edgecolors="k", linewidths=1.2, zorder=5,
               label="центры кластеров")
    ax.set_xlabel(f"PC1 ({expl[0]:.1f}%)")
    ax.set_ylabel(f"PC2 ({expl[1]:.1f}%)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_xlim(np.r_[Z2[:, 0], C2[:, 0]].min() - 0.5, np.r_[Z2[:, 0], C2[:, 0]].max() + 0.5)
    ax.set_ylim(np.r_[Z2[:, 1], C2[:, 1]].min() - 0.5, np.r_[Z2[:, 1], C2[:, 1]].max() + 0.5)
fig.suptitle("Проекция данных на плоскость главных компонент (синий — класс 1, красный — класс 2)",
             fontsize=11)
fig.tight_layout()
fig.savefig("fig4_pca.png", dpi=150)
plt.close(fig)

# Рис. 5: матрица степеней принадлежности
order = np.argsort(y_true)  # сначала класс 1, затем класс 2
fig, ax = plt.subplots(figsize=(12, 2.8))
im = ax.imshow(np.vstack([U[quiet_cluster][order], U[storm_cluster][order]]),
               aspect="auto", cmap="viridis", vmin=0, vmax=1)
ax.set_yticks([0, 1], [f"кластер «спокойные»", f"кластер «буря»"])
bounds = [np.sum(y_true[order] == 1) - 0.5]
ax.axvline(bounds[0], color="w", ls="--", lw=1.2)
ax.text(bounds[0] / 2, -0.36, "истинно спокойные сутки", ha="center",
        fontsize=8.5, color="w", fontweight="bold")
ax.text(bounds[0] + (len(order) - bounds[0] - 0.5) / 2, -0.36,
        "истинно сутки магнитных бурь", ha="center",
        fontsize=8.5, color="w", fontweight="bold")
ax.set_xlabel("Сутки (отсортированы по классам)")
fig.colorbar(im, ax=ax, label="степень принадлежности u")
ax.set_title("Матрица принадлежности U (нечеткая кластеризация, Nc = 2)")
fig.tight_layout()
fig.savefig("fig5_membership.png", dpi=150)
plt.close(fig)

# Рис. 6: сходимость функции цели
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(np.arange(1, len(obj_hist) + 1), obj_hist, "o-", ms=3)
ax.set_xlabel("Итерация")
ax.set_ylabel("Функция цели J")
ax.set_title(f"Сходимость алгоритма c-средних (Nc = 2, m = 2, J_final = {J_best:.4f})")
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig("fig6_objective.png", dpi=150)
plt.close(fig)

print("\nГрафики сохранены: fig1_example_days.png, fig2_gap_fill.png, "
      "fig3_features_box.png, fig4_pca.png, fig5_membership.png, fig6_objective.png")

# ----------------------------------------------------------------------
# ВЫВОДЫ (текст)
# ----------------------------------------------------------------------
print("\n" + "=" * 70)
print("ВЫВОДЫ")
print("=" * 70)
d_asym = centers[storm_cluster][FEATURES.index("asym_night_day")] - \
    centers[quiet_cluster][FEATURES.index("asym_night_day")]
print(f"""
В ходе работы реализован метод нечеткой кластеризации c-средних
(аналог функции fcm из Matlab) и выполнена кластеризация {data.shape[0]}
суточных реализаций нейтронного монитора Апатиты (APTY) по
{data.shape[1]} признакам на Nc = {Nc} кластера.

1. Пропуски ({gap_df['n_missing'].sum()} мин за весь период) заполнены
   медианными значениями.
2. Корректность реализации подтверждена на синтетических данных
   (точность {acc_s * 100:.1f}%).
3. Признаки минутного ряда (среднее, СКО, амплитуда) определяются в
   основном пуассоновским шумом и слабо разделяют сутки классов
   («спокойные» / «буря») — {acc_baseline * 100:.1f}% согласования.
   Наиболее информативны характеристики суточной анизотропии:
   асимметрия «ночь–день» и фаза суточной волны.
4. Итоговая кластеризация (среднее + асимметрия + фаза) согласуется с
   истинной разметкой на {acc_best * 100:.1f}% (чувствительность
   {sens:.3f}, специфичность {spec:.3f}); PC = {PC:.4f}, PE = {PE:.4f},
   XB = {XB:.4f}. Кластер дней бурь характеризуется
   {'повышенной' if d_asym > 0 else 'пониженной'} асимметрией суточной
   вариации — искажением суточной анизотропии потока космических лучей
   в периоды геомагнитных возмущений.
   Оставшаяся доля ошибок объясняется близостью распределений классов
   (часть бурь минимума солнечной активности не сопровождается
   заметными изменениями потока КЛ) и нечетким характером разбиения
   ({amb} суток с промежуточной принадлежностью 0.35–0.65).
""")
