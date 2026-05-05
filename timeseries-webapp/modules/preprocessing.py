import pandas as pd
import numpy as np


def load_and_parse_csv(uploaded_file):
    """CSV 파일을 로드하고 날짜 컬럼을 자동으로 감지합니다."""
    df = pd.read_csv(uploaded_file)
    return df


def detect_date_column(df):
    """날짜 컬럼을 자동으로 감지합니다."""
    date_candidates = []
    for col in df.columns:
        if df[col].dtype == object:
            try:
                parsed = pd.to_datetime(df[col], infer_datetime_format=True)
                if parsed.notna().sum() > len(df) * 0.8:
                    date_candidates.append(col)
            except:
                pass
        elif 'date' in col.lower() or 'time' in col.lower() or 'dt' in col.lower():
            date_candidates.append(col)
    return date_candidates


def detect_value_column(df, date_col):
    """수치형 값 컬럼을 자동으로 감지합니다."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if date_col in numeric_cols:
        numeric_cols.remove(date_col)
    return numeric_cols


def prepare_timeseries(df, date_col, value_col):
    """시계열 데이터를 정리합니다."""
    ts = df[[date_col, value_col]].copy()
    ts[date_col] = pd.to_datetime(ts[date_col])
    ts = ts.sort_values(date_col).reset_index(drop=True)
    ts = ts.rename(columns={date_col: 'ds', value_col: 'y'})
    ts = ts.set_index('ds')
    return ts


def handle_missing_values(ts, method='interpolate'):
    """결측치를 처리합니다."""
    missing_count = ts['y'].isna().sum()
    if method == 'interpolate':
        ts['y'] = ts['y'].interpolate(method='time')
    elif method == 'forward':
        ts['y'] = ts['y'].ffill()
    elif method == 'backward':
        ts['y'] = ts['y'].bfill()
    elif method == 'mean':
        ts['y'] = ts['y'].fillna(ts['y'].mean())
    elif method == 'drop':
        ts = ts.dropna()
    return ts, missing_count


def detect_outliers_iqr(ts, threshold=1.5):
    """IQR 방법으로 이상치를 감지합니다."""
    Q1 = ts['y'].quantile(0.25)
    Q3 = ts['y'].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - threshold * IQR
    upper = Q3 + threshold * IQR
    outliers = ts[(ts['y'] < lower) | (ts['y'] > upper)]
    return outliers, lower, upper


def infer_frequency(ts):
    """시계열 데이터의 빈도를 추론합니다."""
    if len(ts) < 2:
        return 'Unknown'
    diffs = ts.index.to_series().diff().dropna()
    median_diff = diffs.median()
    days = median_diff.days
    if days <= 1:
        return 'Daily'
    elif days <= 7:
        return 'Weekly'
    elif days <= 31:
        return 'Monthly'
    elif days <= 92:
        return 'Quarterly'
    else:
        return 'Yearly'
