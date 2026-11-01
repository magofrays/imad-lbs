"""
Загрузка данных нейтронного монитора Апатиты (APTY) с одноминутным
временным разрешением за даты, заданные вариантом 14 (табл. п.2).

Источник: www.nmdb.eu в настоящий момент недоступен (официальное
объявление на nmdb.eu: доступ к серверам Кильского университета
ограничен, временный интерфейс nest.nmdb.eu закрыт авторизацией),
поэтому данные берутся из базы ИЗМИРАН (cr0.izmiran.ru) —
Всемирного центра данных, откуда поступают данные станции Апатиты
и в NMDB. Канал: "Corrected for pressure data" (скорректированное
на атмосферное давление количество импульсов в минуту) — аналог
канала RCORR в NMDB.

Формирует:
  lb3_data.csv   — все ряды: start_date_time;RCORR_APTY (imp/min)
  lb3_labels.csv — разметка: date;class (1 = спокойный день, 2 = магнитная буря)
"""

import io
import re
import time
import datetime as dt
import urllib.request
import urllib.error

BASE_URL = ("http://cr0.izmiran.ru/scripts/nm64queryD.dll/apty"
            "?PD=0&title=APATITY&dt=0&base=8077&Res=1_min"
            "&y1={y}&y2={y}&m1={m}&m2={m}&d1={d}&d2={d}"
            "&h1=0&h2=23&mn1=0&mn2=59")

# Класс 1 — спокойное состояние (табл. п.2)
QUIET_DATES = [
    "2018-01-03", "2018-01-05", "2018-01-06", "2018-01-07",
    "2018-01-10", "2018-01-11", "2018-01-16", "2018-01-17",
    "2018-02-03", "2018-02-11", "2018-02-13",
    "2018-03-12", "2018-03-13",
    "2018-04-03", "2018-04-06", "2018-04-16", "2018-04-17",
    "2018-05-01", "2018-05-25",
    "2018-06-09", "2018-06-10", "2018-06-11", "2018-06-12",
    "2018-06-15", "2018-06-16", "2018-06-29",
    "2018-07-01", "2018-07-02",
    "2018-09-20",
    "2018-10-17", "2018-10-18", "2018-10-19", "2018-10-24",
    "2018-10-27", "2018-10-28", "2018-10-29", "2018-10-30",
    "2018-11-22", "2018-11-23",
    "2018-12-12", "2018-12-13", "2018-12-14", "2018-12-15", "2018-12-24",
]

# Класс 2 — магнитная буря (табл. п.2)
# Примечание: 2018-10-19 указан в таблице в обеих группах; по планетарному
# индексу Kp (GFZ) этот день крайне спокойный (Kp<=0.7, ap<=3), поэтому он
# отнесён к классу 1 и из класса 2 исключён. Дата 25.05.2018 встречается в
# таблице дважды — учитывается один раз.
STORM_DATES = [
    "2018-01-01", "2018-01-13",
    "2018-03-18", "2018-04-12", "2018-04-20",
    "2018-05-05", "2018-05-06", "2018-05-07",
    "2018-06-01", "2018-06-18", "2018-06-26",
    "2018-07-05",
    "2018-08-20",
    "2018-09-10", "2018-09-11", "2018-09-14", "2018-09-22",
    "2018-10-08", "2018-10-10",
    "2018-11-05",
    "2018-12-08", "2018-12-26", "2018-12-28", "2018-12-29",
    "2019-01-05", "2019-01-25", "2019-02-01", "2019-02-28",
    "2019-03-01", "2019-03-17", "2019-05-14",
    "2019-08-05", "2019-08-31", "2019-09-01", "2019-10-25",
    "2020-03-04", "2020-03-05", "2020-06-08", "2020-09-28", "2020-10-25",
    "2021-03-01", "2021-05-12", "2021-10-12", "2021-11-04",
]

LINE_RE = re.compile(r"^(\d{4})\.(\d{2})\.(\d{2}) (\d{2}):(\d{2})\s+(-?\d+(?:[.,]\d+)?)",
                     re.M)


def fetch_day(session, date_str, retries=3):
    y, m, d = date_str.split("-")
    url = BASE_URL.format(y=y, m=str(int(m)), d=str(int(d)))
    for attempt in range(retries):
        try:
            with session.open(url, timeout=60) as resp:
                raw = resp.read()
            break
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            if attempt == retries - 1:
                raise
            time.sleep(2 * (attempt + 1))
    text = raw.decode("cp1251", errors="replace")
    points = []
    for mm in LINE_RE.finditer(text):
        yy, mo, dd, hh, mi, val = mm.groups()
        # DLL иногда возвращает данные и за последующие дни (до конца
        # месяца) — оставляем только запрошенную дату
        if f"{yy}-{mo}-{dd}" != date_str:
            continue
        points.append((f"{yy}-{mo}-{dd} {hh}:{mi}:00", float(val.replace(",", "."))))
    return points


def main():
    opener = urllib.request.build_opener()
    labels = {ds: 1 for ds in QUIET_DATES}
    labels.update({ds: 2 for ds in STORM_DATES})
    all_dates = sorted(labels)

    out = io.StringIO()
    out.write("start_date_time;RCORR_APTY\n")
    n_days_ok = 0
    report = []
    for i, ds in enumerate(all_dates, 1):
        try:
            pts = fetch_day(opener, ds)
        except Exception as e:
            report.append((ds, labels[ds], 0, f"FAIL: {e}"))
            print(f"[{i}/{len(all_dates)}] {ds}: FAILED ({e})", flush=True)
            continue
        for ts, val in pts:
            out.write(f"{ts};{val:g}\n")
        report.append((ds, labels[ds], len(pts), "ok"))
        n_days_ok += 1
        print(f"[{i}/{len(all_dates)}] {ds}: {len(pts)} rows", flush=True)
        time.sleep(0.4)

    with open("lb3_data.csv", "w", encoding="utf-8") as f:
        f.write(out.getvalue())

    with open("lb3_labels.csv", "w", encoding="utf-8") as f:
        f.write("date;class\n")
        for ds in all_dates:
            f.write(f"{ds};{labels[ds]}\n")

    print("\n=== SUMMARY ===")
    print(f"Days requested: {len(all_dates)}, ok: {n_days_ok}")
    bad = [(ds, c, n, s) for ds, c, n, s in report if n != 1440]
    for ds, c, n, s in bad:
        print(f"  {ds} class={c}: {n} rows ({s})")
    print(f"Days with != 1440 rows: {len(bad)}")
    print("Saved lb3_data.csv and lb3_labels.csv")


if __name__ == "__main__":
    main()
