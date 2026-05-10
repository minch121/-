import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings('ignore')


def train_test_split_ts(ts, test_ratio=0.2):
    n = len(ts)
    split_idx = int(n * (1 - test_ratio))
    train = ts.iloc[:split_idx]
    test = ts.iloc[split_idx:]
    return train, test


def forecast_sma(train, test_steps, forecast_steps, window=None):
    series = train['y']
    if window is None:
        window = max(3, len(series) // 10)
    window = min(window, len(series))

    test_pred = []
    for i in range(test_steps):
        past = list(series) + test_pred[:i]
        val = np.mean(past[-window:])
        test_pred.append(val)

    all_data = list(series) + test_pred
    future_pred = []
    for i in range(forecast_steps):
        val = np.mean(all_data[-window:])
        future_pred.append(val)
        all_data.append(val)

    return np.array(test_pred), np.array(future_pred), {'window': window}


def forecast_exponential_smoothing(train, test_steps, forecast_steps, alpha=None):
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


def _detect_seasonal_period(series):
    """데이터 빈도를 자동 감지해서 계절 주기 반환"""
    try:
        diffs = series.index.to_series().diff().dropna()
        median_days = diffs.median().days
        if median_days <= 1:
            return 7        # 일별 → 주 단위 계절성 (365는 데이터가 너무 많이 필요)
        elif median_days <= 7:
            return 52       # 주별 → 연 단위
        elif median_days <= 31:
            return 12       # 월별 → 연 단위
        elif median_days <= 92:
            return 4        # 분기별 → 연 단위
        else:
            return None
    except:
        return None


def forecast_holt_winters(train, test_steps, forecast_steps, seasonal_periods=None):
    """Holt-Winters 예측 - 빈도 자동 감지 + 발산 방지"""
    series = train['y']
    series_min = series.min()
    series_max = series.max()
    margin = (series_max - series_min) * 2

    # 계절 주기 자동 감지
    if seasonal_periods is None:
        seasonal_periods = _detect_seasonal_period(series)

    # 데이터가 주기의 2배 이상일 때만 계절성 적용
    use_seasonal = (
        seasonal_periods is not None and
        len(series) >= 2 * seasonal_periods
    )

    try:
        if use_seasonal:
            model = ExponentialSmoothing(
                series,
                trend='add',
                seasonal='add',
                seasonal_periods=seasonal_periods,
                initialization_method='estimated'
            )
        else:
            model = ExponentialSmoothing(
                series,
                trend='add',
                seasonal=None,
                initialization_method='estimated'
            )

        fitted = model.fit(optimized=True)
        all_forecast = fitted.forecast(test_steps + forecast_steps)

        # 발산 방지: 범위 벗어나면 ES로 대체
        if (all_forecast.min() < series_min - margin or
                all_forecast.max() > series_max + margin):
            return forecast_exponential_smoothing(train, test_steps, forecast_steps)

        test_pred = all_forecast[:test_steps].values
        future_pred = all_forecast[test_steps:].values
        params = {
            'alpha': round(fitted.params.get('smoothing_level', 0), 4),
            'beta': round(fitted.params.get('smoothing_trend', 0), 4),
            'seasonal_periods': seasonal_periods if use_seasonal else None
        }
        return test_pred, future_pred, params

    except Exception:
        return forecast_exponential_smoothing(train, test_steps, forecast_steps)


def forecast_arima(train, test_steps, forecast_steps, order=None):
    series = train['y']
    if order is None:
        order = auto_arima_simple(series)

    try:
        model = ARIMA(series, order=order)
        fitted = model.fit()
        all_forecast = fitted.forecast(steps=test_steps + forecast_steps)
        test_pred = all_forecast[:test_steps].values
        future_pred = all_forecast[test_steps:].values

        # ARIMA도 발산 방지
        series_min = series.min()
        series_max = series.max()
        margin = (series_max - series_min) * 2
        if (all_forecast.min() < series_min - margin or
                all_forecast.max() > series_max + margin):
            return forecast_exponential_smoothing(train, test_steps, forecast_steps)

        return test_pred, future_pred, {
            'order': order,
            'aic': round(fitted.aic, 2),
            'bic': round(fitted.bic, 2)
        }
    except Exception:
        return forecast_exponential_smoothing(train, test_steps, forecast_steps)


def auto_arima_simple(series):
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
        return forecast_exponential_smoothing(train, test_steps, forecast_steps)


def generate_future_dates(ts, steps):
    last_date = ts.index[-1]
    diffs = ts.index.to_series().diff().dropna()
    freq = diffs.median()
    future_dates = [last_date + freq * (i + 1) for i in range(steps)]
    return future_dates