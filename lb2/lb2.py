import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
