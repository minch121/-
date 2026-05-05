import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose
import warnings
warnings.filterwarnings('ignore')


def adf_test(series):
    """ADF 정상성 검정"""
    result = adfuller(series.dropna(), autolag='AIC')
    return {
        'test_statistic': result[0],
        'p_value': result[1],
        'lags_used': result[2],
        'n_obs': result[3],
        'critical_values': result[4],
        'is_stationary': result[1] < 0.05
    }


def kpss_test(series):
    """KPSS 정상성 검정"""
    try:
        result = kpss(series.dropna(), regression='c', nlags='auto')
        return {
            'test_statistic': result[0],
            'p_value': result[1],
            'lags_used': result[2],
            'critical_values': result[3],
            'is_stationary': result[1] > 0.05
        }
    except:
        return None


def compute_acf_pacf(series, nlags=40):
    """ACF와 PACF를 계산합니다."""
    n = len(series.dropna())
    nlags = min(nlags, n // 2 - 1)
    acf_values = acf(series.dropna(), nlags=nlags, fft=True)
    pacf_values = pacf(series.dropna(), nlags=nlags, method='ols')
    confidence_interval = 1.96 / np.sqrt(n)
    return acf_values, pacf_values, confidence_interval, nlags


def decompose_timeseries(ts, period=None, model='additive'):
    """시계열을 분해합니다."""
    series = ts['y'].dropna()
    n = len(series)

    if period is None:
        # 자동 주기 추론
        diffs = ts.index.to_series().diff().dropna()
        median_days = diffs.median().days
        if median_days <= 1:
            period = 7      # daily → weekly seasonality
        elif median_days <= 7:
            period = 52     # weekly → yearly
        elif median_days <= 31:
            period = 12     # monthly → yearly
        else:
            period = 4      # quarterly → yearly

    if n < 2 * period:
        period = max(2, n // 3)

    try:
        decomp = seasonal_decompose(series, model=model, period=period, extrapolate_trend='freq')
        return decomp, period
    except Exception as e:
        return None, period


def get_descriptive_stats(series):
    """기술통계량을 계산합니다."""
    return {
        '관측값 수': len(series),
        '평균': series.mean(),
        '중앙값': series.median(),
        '표준편차': series.std(),
        '최솟값': series.min(),
        '최댓값': series.max(),
        '왜도': series.skew(),
        '첨도': series.kurtosis(),
        '결측치': series.isna().sum(),
    }


def suggest_model(adf_result, series):
    """분석 결과를 기반으로 모델을 추천합니다."""
    suggestions = []
    reasons = []

    is_stationary = adf_result['is_stationary']

    if not is_stationary:
        suggestions.append('ARIMA')
        reasons.append('비정상 시계열 → 차분이 포함된 ARIMA 권장')
    else:
        suggestions.append('ARIMA')
        reasons.append('정상 시계열 → ARIMA(p,0,q) 적합')

    # 계절성 판단 (단순 방식)
    if len(series) >= 24:
        suggestions.append('SARIMA')
        reasons.append('데이터 길이 충분 → 계절성 SARIMA 시도 가능')
        suggestions.append('Holt-Winters')
        reasons.append('추세 + 계절성 → Holt-Winters 평활법 적합')

    suggestions.append('Prophet')
    reasons.append('트렌드/계절성 자동 처리 → Prophet 범용 추천')

    return suggestions, reasons
