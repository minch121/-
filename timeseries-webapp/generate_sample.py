import pandas as pd
import numpy as np

np.random.seed(42)
dates = pd.date_range(start='2020-01-01', periods=365, freq='D')
trend = np.linspace(100, 150, 365)
seasonal = 10 * np.sin(2 * np.pi * np.arange(365) / 365)
noise = np.random.normal(0, 3, 365)
values = trend + seasonal + noise

df = pd.DataFrame({'date': dates, 'value': values.round(2)})
df.to_csv('/home/claude/timeseries-webapp/sample_data/sample_daily.csv', index=False)

dates2 = pd.date_range(start='2018-01-01', periods=60, freq='MS')
trend2 = np.linspace(200, 300, 60)
seasonal2 = 20 * np.sin(2 * np.pi * np.arange(60) / 12)
noise2 = np.random.normal(0, 5, 60)
values2 = trend2 + seasonal2 + noise2
df2 = pd.DataFrame({'date': dates2, 'sales': values2.round(2)})
df2.to_csv('/home/claude/timeseries-webapp/sample_data/sample_monthly.csv', index=False)

print("샘플 데이터 생성 완료")
