import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings('ignore')


def train_test_split_ts(ts, test_ratio=0.2):
    """시계열 데이터를 train/test로 분리합니다."""
    n = len(ts)
    split_idx = int(n * (1 - test_ratio))
    train = ts.iloc[:split_idx]
    test = ts.iloc[split_idx:]
    return train, test


def forecast_sma(train, test_steps, forecast_steps, window=None):
    """단순이동평균 예측"""
    series = train['y']
    if window is None:
        window = max(3, len(series) // 10)
    window = min(window, len(series))

    # 테스트셋 예측
    test_pred = []
    for i in range(test_steps):
        start = max(0, len(series) - window + i)
        end = len(series) + i
        if i == 0:
            val = series[-window:].mean()
        else:
            past = list(series) + test_pred[:i]
            val = np.mean(past[-window:])
        test_pred.append(val)

    # 미래 예측
    all_data = list(series) + test_pred
    future_pred = []
    for i in range(forecast_steps):
        val = np.mean(all_data[-window:])
        future_pred.append(val)
        all_data.append(val)

    return np.array(test_pred), np.array(future_pred), {'window': window}


def forecast_exponential_smoothing(train, test_steps, forecast_steps, alpha=None):
    """지수평활 예측 (단순)"""
    series = train['y']
    if alpha is None:
        model = SimpleExpSmoothing(series, initialization_method='estimated')
        fitted = model.fit(optimized=True)
        alpha = fitted.params['smoothing_level']
    else:
        model = SimpleExpSmoothing(series, initialization_method='estimated')
        fitted = model.fit(smoothing_level=alpha, optimized=False)

    test_pred = fitted.forecast(test_steps).values
    future_pred = fitted.forecast(test_steps + forecast_steps).values[-forecast_steps:]
    return test_pred, future_pred, {'alpha': round(alpha, 4)}


def forecast_holt_winters(train, test_steps, forecast_steps, seasonal_periods=None):
    """Holt-Winters 예측"""
    series = train['y']
    try:
        if seasonal_periods is None:
            n = len(series)
            if n >= 24:
                seasonal_periods = 12
            elif n >= 14:
                seasonal_periods = 7
            else:
                seasonal_periods = None

        if seasonal_periods and len(series) >= 2 * seasonal_periods:
            model = ExponentialSmoothing(
                series, trend='add', seasonal='add',
                seasonal_periods=seasonal_periods,
                initialization_method='estimated'
            )
        else:
            model = ExponentialSmoothing(
                series, trend='add', seasonal=None,
                initialization_method='estimated'
            )

        fitted = model.fit(optimized=True, method='ls')
        all_forecast = fitted.forecast(test_steps + forecast_steps)

        # 발산 방지: 예측값이 학습 데이터 범위의 3배 초과하면 ES로 대체
        series_min, series_max = series.min(), series.max()
        margin = (series_max - series_min) * 3
        if all_forecast.min() < series_min - margin or all_forecast.max() > series_max + margin:
            return forecast_exponential_smoothing(train, test_steps, forecast_steps)

        test_pred  = all_forecast[:test_steps].values
        future_pred = all_forecast[test_steps:].values
        params = {
            'alpha': round(fitted.params.get('smoothing_level', 0), 4),
            'beta':  round(fitted.params.get('smoothing_trend', 0), 4),
            'seasonal_periods': seasonal_periods
        }
        return test_pred, future_pred, params

    except Exception as e:
        return forecast_exponential_smoothing(train, test_steps, forecast_steps)


def forecast_arima(train, test_steps, forecast_steps, order=None):
    """ARIMA 예측"""
    series = train['y']

    if order is None:
        order = auto_arima_simple(series)

    try:
        model = ARIMA(series, order=order)
        fitted = model.fit()
        all_forecast = fitted.forecast(steps=test_steps + forecast_steps)
        test_pred = all_forecast[:test_steps].values
        future_pred = all_forecast[test_steps:].values
        return test_pred, future_pred, {
            'order': order,
            'aic': round(fitted.aic, 2),
            'bic': round(fitted.bic, 2)
        }
    except Exception as e:
        return forecast_holt_winters(train, test_steps, forecast_steps)


def auto_arima_simple(series):
    """간단한 ARIMA 파라미터 선택 (AIC 기반)"""
    from statsmodels.tsa.stattools import adfuller
    adf_p = adfuller(series.dropna())[1]
    d = 0 if adf_p < 0.05 else 1

    best_aic = np.inf
    best_order = (1, d, 1)

    for p in range(0, 4):
        for q in range(0, 4):
            try:
                m = ARIMA(series, order=(p, d, q))
                f = m.fit()
                if f.aic < best_aic:
                    best_aic = f.aic
                    best_order = (p, d, q)
            except:
                pass
    return best_order


def forecast_prophet(train, test_steps, forecast_steps):
    """Prophet 예측"""
    try:
        from prophet import Prophet
        df_train = train.reset_index().rename(columns={'ds': 'ds', 'y': 'y'})
        df_train.columns = ['ds', 'y']

        model = Prophet(yearly_seasonality='auto', weekly_seasonality='auto', daily_seasonality=False)
        model.fit(df_train)

        future = model.make_future_dataframe(periods=test_steps + forecast_steps)
        forecast = model.predict(future)

        test_pred = forecast['yhat'].values[-(test_steps + forecast_steps):-forecast_steps]
        future_pred = forecast['yhat'].values[-forecast_steps:]

        return test_pred, future_pred, {'model': 'Prophet (Facebook)'}
    except ImportError:
        return forecast_holt_winters(train, test_steps, forecast_steps)


def generate_future_dates(ts, steps):
    """미래 날짜를 생성합니다."""
    last_date = ts.index[-1]
    diffs = ts.index.to_series().diff().dropna()
    freq = diffs.median()
    future_dates = [last_date + freq * (i + 1) for i in range(steps)]
    return future_dates
