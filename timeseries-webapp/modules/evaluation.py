import numpy as np
import pandas as pd


def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def mean_absolute_percentage_error(y_true, y_pred):
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
    return 1 - ss_res / ss_tot


def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def evaluate_model(y_true, y_pred, model_name):
    """모든 평가지표를 계산합니다."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    return {
        'Model': model_name,
        'MAE': round(mean_absolute_error(y_true, y_pred), 4),
        'RMSE': round(root_mean_squared_error(y_true, y_pred), 4),
        'MAPE (%)': round(mean_absolute_percentage_error(y_true, y_pred), 4),
        'R²': round(r2_score(y_true, y_pred), 4),
        'MSE': round(mean_squared_error(y_true, y_pred), 4),
    }


def get_best_model(results_df):
    """RMSE 기준으로 최적 모델을 반환합니다."""
    if results_df.empty:
        return None
    return results_df.loc[results_df['RMSE'].idxmin(), 'Model']


def rank_models(results_df):
    """모델을 RMSE 기준으로 순위를 매깁니다."""
    df = results_df.copy()
    df['Rank'] = df['RMSE'].rank().astype(int)
    return df.sort_values('Rank')
