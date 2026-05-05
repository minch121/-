import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

from modules.preprocessing import (
    load_and_parse_csv, detect_date_column, detect_value_column,
    prepare_timeseries, handle_missing_values, detect_outliers_iqr, infer_frequency
)
from modules.analysis import (
    adf_test, kpss_test, compute_acf_pacf, decompose_timeseries,
    get_descriptive_stats, suggest_model
)
from modules.forecasting import (
    train_test_split_ts, forecast_sma, forecast_exponential_smoothing,
    forecast_holt_winters, forecast_arima, forecast_prophet, generate_future_dates
)
from modules.evaluation import evaluate_model, get_best_model, rank_models

# ─── 페이지 설정 ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="시계열 분석 & 예측 웹앱",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CSS ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1f77b4;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 0.95rem;
        color: #666;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        border-left: 4px solid #1f77b4;
    }
    .stAlert { border-radius: 8px; }
    .section-title {
        font-size: 1.3rem;
        font-weight: 600;
        color: #333;
        border-bottom: 2px solid #1f77b4;
        padding-bottom: 5px;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .best-model-badge {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# ─── 헤더 ────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-header">📈 시계열 분석 & 예측 웹앱</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">CSV 파일을 업로드하면 자동으로 시계열 분석 및 예측을 수행합니다</div>', unsafe_allow_html=True)

# ─── 사이드바 ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/combo-chart.png", width=60)
    st.title("⚙️ 설정")

    st.markdown("---")
    st.markdown("### 📂 데이터 업로드")

    use_sample = st.checkbox("📦 샘플 데이터 사용", value=False)
    if use_sample:
        sample_choice = st.selectbox("샘플 선택", ["sample_daily.csv", "sample_monthly.csv"])

    uploaded_file = st.file_uploader("CSV 파일 업로드", type=['csv'])

    st.markdown("---")
    st.markdown("### 🔬 분석 설정")

    missing_method = st.selectbox(
        "결측치 처리 방법",
        ['interpolate', 'forward', 'backward', 'mean', 'drop'],
        format_func=lambda x: {
            'interpolate': '보간법', 'forward': '전방채움',
            'backward': '후방채움', 'mean': '평균채움', 'drop': '제거'
        }[x]
    )

    decomp_model = st.radio("분해 모델", ['additive', 'multiplicative'],
                            format_func=lambda x: '가법 모델' if x == 'additive' else '승법 모델')

    st.markdown("---")
    st.markdown("### 🔮 예측 설정")

    forecast_horizon = st.slider("⏱️ 예측 시평 (Forecast Horizon)", 7, 365, 30, step=1,
                                  help="미래 몇 스텝을 예측할지 설정합니다")

    test_ratio = st.slider("테스트셋 비율", 0.1, 0.4, 0.2, step=0.05,
                            format="%.0f%%",
                            help="전체 데이터 중 평가에 사용할 비율")

    st.markdown("---")
    st.markdown("### 🤖 예측 모델 선택")

    model_sma = st.checkbox("단순이동평균 (SMA)", value=True)
    sma_window = st.slider("SMA 윈도우 크기", 3, 60, 7, disabled=not model_sma)

    model_es = st.checkbox("지수평활 (ES)", value=True)
    es_alpha = st.slider("평활 계수 α", 0.01, 1.0, 0.3, step=0.01, disabled=not model_es)

    model_hw = st.checkbox("Holt-Winters", value=True)

    model_arima = st.checkbox("ARIMA (자동)", value=True)
    arima_manual = st.checkbox("ARIMA 수동 설정", value=False, disabled=not model_arima)
    if arima_manual and model_arima:
        col1, col2, col3 = st.columns(3)
        arima_p = col1.number_input("p", 0, 5, 1)
        arima_d = col2.number_input("d", 0, 2, 0)
        arima_q = col3.number_input("q", 0, 5, 1)
        arima_order = (arima_p, arima_d, arima_q)
    else:
        arima_order = None

    model_prophet = st.checkbox("Prophet", value=False)

    st.markdown("---")
    st.caption("산업데이터공학과 시계열분석\n1차 프로젝트")


# ─── 데이터 로드 ──────────────────────────────────────────────────────────────
def get_data():
    if use_sample:
        import os
        path = f"sample_data/{sample_choice}"
        if os.path.exists(path):
            return pd.read_csv(path)
    if uploaded_file is not None:
        return load_and_parse_csv(uploaded_file)
    return None


df_raw = get_data()

# ─── 메인 탭 ─────────────────────────────────────────────────────────────────
if df_raw is None:
    # 랜딩 페이지
    st.info("👈 **사이드바에서 CSV 파일을 업로드하거나 샘플 데이터를 선택하세요.**")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div class="metric-card">
        <h3>📂 1. 데이터 업로드</h3>
        <p>단변량 시계열 CSV 파일을 업로드합니다. 날짜 컬럼과 값 컬럼을 자동으로 감지합니다.</p>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-card">
        <h3>🔬 2. 자동 분석</h3>
        <p>정상성 검정(ADF/KPSS), 시계열 분해, ACF/PACF 분석을 자동으로 수행합니다.</p>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-card">
        <h3>🔮 3. 예측 수행</h3>
        <p>SMA, 지수평활, Holt-Winters, ARIMA, Prophet 등 다양한 모델로 예측합니다.</p>
        </div>""", unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <div class="metric-card">
        <h3>📊 4. 성능 평가</h3>
        <p>MAE, RMSE, MAPE, R² 지표와 인터랙티브 대시보드로 예측 성능을 평가합니다.</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### 📋 CSV 형식 예시")
    example = pd.DataFrame({
        'date': ['2024-01-01', '2024-01-02', '2024-01-03'],
        'value': [120.5, 125.3, 118.7]
    })
    st.dataframe(example, use_container_width=True)

else:
    # ─── 컬럼 설정 ───────────────────────────────────────────────────────────
    date_candidates = detect_date_column(df_raw)
    numeric_candidates = detect_value_column(df_raw, date_candidates[0] if date_candidates else '')

    col_config1, col_config2 = st.columns(2)
    with col_config1:
        if date_candidates:
            date_col = st.selectbox("📅 날짜 컬럼", date_candidates, index=0)
        else:
            date_col = st.selectbox("📅 날짜 컬럼 (직접 선택)", df_raw.columns.tolist())
    with col_config2:
        value_col = st.selectbox("📊 값 컬럼", numeric_candidates if numeric_candidates else df_raw.columns.tolist())

    # ─── 데이터 전처리 ────────────────────────────────────────────────────────
    try:
        ts = prepare_timeseries(df_raw, date_col, value_col)
        ts, missing_count = handle_missing_values(ts, method=missing_method)
        freq_label = infer_frequency(ts)
        outliers, lower_bound, upper_bound = detect_outliers_iqr(ts)
    except Exception as e:
        st.error(f"데이터 처리 오류: {e}")
        st.stop()

    # ─── 탭 구성 ─────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 데이터 개요", "🔬 시계열 분석", "🔮 예측 수행", "📈 성능 평가 대시보드"
    ])

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 1: 데이터 개요
    # ══════════════════════════════════════════════════════════════════════════
    with tab1:
        st.markdown('<div class="section-title">📋 데이터 기본 정보</div>', unsafe_allow_html=True)

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("📏 관측값 수", f"{len(ts):,}")
        m2.metric("📅 시작일", str(ts.index[0].date()))
        m3.metric("📅 종료일", str(ts.index[-1].date()))
        m4.metric("⏱️ 빈도", freq_label)
        m5.metric("❓ 결측치", f"{missing_count}개")

        col_l, col_r = st.columns([2, 1])
        with col_l:
            st.markdown('<div class="section-title">📈 시계열 원본 데이터</div>', unsafe_allow_html=True)
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=ts.index, y=ts['y'],
                mode='lines', name='원본 데이터',
                line=dict(color='#1f77b4', width=1.5)
            ))
            if len(outliers) > 0:
                fig.add_trace(go.Scatter(
                    x=outliers.index, y=outliers['y'],
                    mode='markers', name='이상치',
                    marker=dict(color='red', size=8, symbol='x')
                ))
            fig.update_layout(
                height=350, margin=dict(l=0, r=0, t=30, b=0),
                hovermode='x unified',
                legend=dict(orientation='h', y=1.05)
            )
            st.plotly_chart(fig, use_container_width=True)

        with col_r:
            st.markdown('<div class="section-title">📊 기술통계</div>', unsafe_allow_html=True)
            stats = get_descriptive_stats(ts['y'])
            stats_df = pd.DataFrame({
                '항목': list(stats.keys()),
                '값': [f"{v:,.4f}" if isinstance(v, float) else str(v) for v in stats.values()]
            })
            st.dataframe(stats_df, use_container_width=True, hide_index=True)

        # 분포
        st.markdown('<div class="section-title">📉 데이터 분포</div>', unsafe_allow_html=True)
        col_h1, col_h2 = st.columns(2)
        with col_h1:
            fig_hist = px.histogram(ts, x='y', nbins=40, title='값 분포 (히스토그램)',
                                    color_discrete_sequence=['#1f77b4'])
            fig_hist.update_layout(height=280, margin=dict(l=0,r=0,t=40,b=0))
            st.plotly_chart(fig_hist, use_container_width=True)
        with col_h2:
            fig_box = px.box(ts, y='y', title='박스플롯',
                             color_discrete_sequence=['#ff7f0e'])
            fig_box.update_layout(height=280, margin=dict(l=0,r=0,t=40,b=0))
            st.plotly_chart(fig_box, use_container_width=True)

        st.markdown('<div class="section-title">🗂️ 원본 데이터 미리보기</div>', unsafe_allow_html=True)
        st.dataframe(df_raw.head(20), use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 2: 시계열 분석
    # ══════════════════════════════════════════════════════════════════════════
    with tab2:
        # 정상성 검정
        st.markdown('<div class="section-title">🧪 정상성 검정</div>', unsafe_allow_html=True)
        adf_result = adf_test(ts['y'])
        kpss_result = kpss_test(ts['y'])

        col_adf, col_kpss = st.columns(2)
        with col_adf:
            adf_color = "🟢" if adf_result['is_stationary'] else "🔴"
            verdict = "**정상(Stationary)**" if adf_result['is_stationary'] else "**비정상(Non-Stationary)**"
            st.markdown(f"#### {adf_color} ADF 검정 결과: {verdict}")
            st.markdown(f"""
            | 항목 | 값 |
            |---|---|
            | 검정통계량 | `{adf_result['test_statistic']:.4f}` |
            | p-value | `{adf_result['p_value']:.4f}` |
            | 기준 (p < 0.05) | 정상 판정 |
            """)
            for key, val in adf_result['critical_values'].items():
                st.caption(f"임계값 {key}: {val:.4f}")

        with col_kpss:
            if kpss_result:
                kpss_color = "🟢" if kpss_result['is_stationary'] else "🔴"
                verdict_k = "**정상(Stationary)**" if kpss_result['is_stationary'] else "**비정상(Non-Stationary)**"
                st.markdown(f"#### {kpss_color} KPSS 검정 결과: {verdict_k}")
                st.markdown(f"""
                | 항목 | 값 |
                |---|---|
                | 검정통계량 | `{kpss_result['test_statistic']:.4f}` |
                | p-value | `{kpss_result['p_value']:.4f}` |
                | 기준 (p > 0.05) | 정상 판정 |
                """)

        # 모델 추천
        suggestions, reasons = suggest_model(adf_result, ts['y'])
        st.markdown('<div class="section-title">💡 추천 모델</div>', unsafe_allow_html=True)
        for s, r in zip(suggestions, reasons):
            st.markdown(f"- **{s}**: {r}")

        # ACF / PACF
        st.markdown('<div class="section-title">📉 ACF / PACF</div>', unsafe_allow_html=True)
        acf_vals, pacf_vals, ci, nlags = compute_acf_pacf(ts['y'])

        fig_acfpacf = make_subplots(rows=1, cols=2,
                                     subplot_titles=['자기상관함수 (ACF)', '편자기상관함수 (PACF)'])
        lags = list(range(len(acf_vals)))
        for i, val in enumerate(acf_vals):
            fig_acfpacf.add_trace(go.Bar(x=[i], y=[val], marker_color='#1f77b4',
                                          showlegend=False), row=1, col=1)
        for i, val in enumerate(pacf_vals):
            fig_acfpacf.add_trace(go.Bar(x=[i], y=[val], marker_color='#ff7f0e',
                                          showlegend=False), row=1, col=2)
        for col_idx in [1, 2]:
            fig_acfpacf.add_hline(y=ci, line_dash="dash", line_color="red", opacity=0.5, row=1, col=col_idx)
            fig_acfpacf.add_hline(y=-ci, line_dash="dash", line_color="red", opacity=0.5, row=1, col=col_idx)

        fig_acfpacf.update_layout(height=320, margin=dict(l=0,r=0,t=40,b=0))
        st.plotly_chart(fig_acfpacf, use_container_width=True)
        st.caption("빨간 점선: 95% 신뢰구간 (유의한 자기상관 기준선)")

        # 시계열 분해
        st.markdown('<div class="section-title">🔧 시계열 분해 (Decomposition)</div>', unsafe_allow_html=True)
        with st.spinner("시계열 분해 중..."):
            decomp, period_used = decompose_timeseries(ts, model=decomp_model)

        if decomp:
            st.caption(f"사용된 주기(period): {period_used}")
            fig_decomp = make_subplots(rows=4, cols=1,
                                        subplot_titles=['원본 (Observed)', '추세 (Trend)',
                                                        '계절성 (Seasonal)', '잔차 (Residual)'],
                                        shared_xaxes=True, vertical_spacing=0.06)
            colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728']
            components = [decomp.observed, decomp.trend, decomp.seasonal, decomp.resid]
            for i, (comp, color) in enumerate(zip(components, colors)):
                fig_decomp.add_trace(
                    go.Scatter(x=comp.index, y=comp.values, mode='lines',
                               line=dict(color=color, width=1.2), showlegend=False),
                    row=i+1, col=1
                )
            fig_decomp.update_layout(height=700, margin=dict(l=0,r=0,t=40,b=0))
            st.plotly_chart(fig_decomp, use_container_width=True)
        else:
            st.warning("데이터가 짧아 분해를 수행할 수 없습니다.")

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 3: 예측 수행
    # ══════════════════════════════════════════════════════════════════════════
    with tab3:
        st.markdown(f'<div class="section-title">🔮 예측 시평: {forecast_horizon} 스텝 앞</div>',
                    unsafe_allow_html=True)

        if not any([model_sma, model_es, model_hw, model_arima, model_prophet]):
            st.warning("사이드바에서 예측 모델을 하나 이상 선택하세요.")
        else:
            train, test = train_test_split_ts(ts, test_ratio=test_ratio)
            test_steps = len(test)
            future_dates = generate_future_dates(ts, forecast_horizon)

            st.info(f"🔹 학습 데이터: {len(train)}개 | 테스트 데이터: {len(test)}개 | 예측: {forecast_horizon}스텝")

            results = {}
            forecasts_future = {}
            eval_rows = []

            progress = st.progress(0, text="예측 모델 실행 중...")
            model_list = []
            if model_sma: model_list.append('SMA')
            if model_es: model_list.append('ES')
            if model_hw: model_list.append('Holt-Winters')
            if model_arima: model_list.append('ARIMA')
            if model_prophet: model_list.append('Prophet')

            for idx, mname in enumerate(model_list):
                progress.progress((idx+1)/len(model_list), text=f"{mname} 실행 중...")
                try:
                    if mname == 'SMA':
                        tp, fp, params = forecast_sma(train, test_steps, forecast_horizon, sma_window)
                    elif mname == 'ES':
                        tp, fp, params = forecast_exponential_smoothing(train, test_steps, forecast_horizon, es_alpha)
                    elif mname == 'Holt-Winters':
                        tp, fp, params = forecast_holt_winters(train, test_steps, forecast_horizon)
                    elif mname == 'ARIMA':
                        tp, fp, params = forecast_arima(train, test_steps, forecast_horizon, arima_order)
                    elif mname == 'Prophet':
                        tp, fp, params = forecast_prophet(train, test_steps, forecast_horizon)

                    results[mname] = {'test_pred': tp, 'params': params}
                    forecasts_future[mname] = fp
                    eval_row = evaluate_model(test['y'].values, tp, mname)
                    eval_rows.append(eval_row)
                except Exception as e:
                    st.warning(f"{mname} 실행 오류: {e}")

            progress.empty()

            if not results:
                st.error("예측에 실패했습니다.")
            else:
                eval_df = pd.DataFrame(eval_rows)
                best_model = get_best_model(eval_df)

                # ── 메인 예측 차트
                st.markdown('<div class="section-title">📈 예측 결과 (테스트셋)</div>',
                            unsafe_allow_html=True)

                colors_map = {
                    'SMA': '#ff7f0e', 'ES': '#2ca02c',
                    'Holt-Winters': '#9467bd', 'ARIMA': '#d62728', 'Prophet': '#e377c2'
                }

                fig_pred = go.Figure()
                fig_pred.add_trace(go.Scatter(
                    x=train.index, y=train['y'], mode='lines',
                    name='학습 데이터', line=dict(color='#1f77b4', width=1.5)
                ))
                fig_pred.add_trace(go.Scatter(
                    x=test.index, y=test['y'], mode='lines',
                    name='실제값 (테스트)', line=dict(color='black', width=2, dash='dot')
                ))
                for mname, res in results.items():
                    is_best = mname == best_model
                    fig_pred.add_trace(go.Scatter(
                        x=test.index, y=res['test_pred'], mode='lines',
                        name=f"{mname}" + (" ⭐" if is_best else ""),
                        line=dict(color=colors_map.get(mname, '#333'),
                                  width=3 if is_best else 1.5,
                                  dash='solid' if is_best else 'dash')
                    ))
                # 구분선
                fig_pred.add_vline(
                    x=test.index[0], line_dash="dash", line_color="gray",
                    annotation_text="테스트 시작", annotation_position="top right"
                )
                fig_pred.update_layout(
                    height=420, hovermode='x unified',
                    margin=dict(l=0,r=0,t=30,b=0),
                    legend=dict(orientation='h', y=-0.15)
                )
                st.plotly_chart(fig_pred, use_container_width=True)

                # ── 미래 예측 차트
                st.markdown(f'<div class="section-title">🚀 미래 예측 ({forecast_horizon}스텝)</div>',
                            unsafe_allow_html=True)

                fig_future = go.Figure()
                fig_future.add_trace(go.Scatter(
                    x=ts.index[-min(100, len(ts)):],
                    y=ts['y'].values[-min(100, len(ts)):],
                    mode='lines', name='실제값 (최근)',
                    line=dict(color='#1f77b4', width=2)
                ))
                for mname, fp in forecasts_future.items():
                    is_best = mname == best_model
                    fig_future.add_trace(go.Scatter(
                        x=future_dates, y=fp, mode='lines+markers',
                        name=f"{mname}" + (" ⭐" if is_best else ""),
                        line=dict(color=colors_map.get(mname, '#333'),
                                  width=3 if is_best else 1.5),
                        marker=dict(size=4)
                    ))
                fig_future.add_vline(
                    x=ts.index[-1], line_dash="dash", line_color="green",
                    annotation_text="현재", annotation_position="top right"
                )
                fig_future.update_layout(
                    height=380, hovermode='x unified',
                    margin=dict(l=0,r=0,t=30,b=0),
                    legend=dict(orientation='h', y=-0.18)
                )
                st.plotly_chart(fig_future, use_container_width=True)

                # 예측값 테이블
                with st.expander("📋 미래 예측값 테이블 보기"):
                    future_df = pd.DataFrame({'날짜': future_dates})
                    for mname, fp in forecasts_future.items():
                        future_df[mname] = fp.round(4)
                    st.dataframe(future_df, use_container_width=True)

                # 모델 파라미터
                with st.expander("🔧 모델 파라미터"):
                    for mname, res in results.items():
                        st.markdown(f"**{mname}**: {res['params']}")

                # 세션에 저장
                st.session_state['eval_df'] = eval_df
                st.session_state['results'] = results
                st.session_state['test'] = test
                st.session_state['best_model'] = best_model
                st.session_state['forecasts_future'] = forecasts_future
                st.session_state['future_dates'] = future_dates

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 4: 성능 평가 대시보드
    # ══════════════════════════════════════════════════════════════════════════
    with tab4:
        if 'eval_df' not in st.session_state:
            st.info("👈 먼저 **예측 수행** 탭에서 예측을 실행하세요.")
        else:
            eval_df = st.session_state['eval_df']
            results = st.session_state['results']
            test = st.session_state['test']
            best_model = st.session_state['best_model']

            st.markdown(f'<div class="section-title">🏆 최적 모델: <span class="best-model-badge">{best_model} (RMSE 기준)</span></div>',
                        unsafe_allow_html=True)

            # 평가지표 테이블
            st.markdown('<div class="section-title">📊 모델별 성능 지표 비교</div>', unsafe_allow_html=True)
            ranked = rank_models(eval_df)
            styled = ranked.style.highlight_min(subset=['MAE','RMSE','MAPE (%)','MSE'],
                                                 color='#d4edda').highlight_max(
                                                     subset=['R²'], color='#d4edda')
            st.dataframe(ranked, use_container_width=True, hide_index=True)

            # 레이더 차트 + 막대 차트
            col_r1, col_r2 = st.columns(2)

            with col_r1:
                st.markdown("#### RMSE 비교")
                fig_bar = px.bar(eval_df, x='Model', y='RMSE', color='Model',
                                  color_discrete_sequence=px.colors.qualitative.Set2,
                                  title='RMSE (낮을수록 좋음)')
                fig_bar.update_layout(height=300, margin=dict(l=0,r=0,t=40,b=0), showlegend=False)
                st.plotly_chart(fig_bar, use_container_width=True)

            with col_r2:
                st.markdown("#### MAPE (%) 비교")
                fig_mape = px.bar(eval_df, x='Model', y='MAPE (%)', color='Model',
                                   color_discrete_sequence=px.colors.qualitative.Pastel,
                                   title='MAPE % (낮을수록 좋음)')
                fig_mape.update_layout(height=300, margin=dict(l=0,r=0,t=40,b=0), showlegend=False)
                st.plotly_chart(fig_mape, use_container_width=True)

            # R² 게이지 차트
            st.markdown('<div class="section-title">📐 R² 결정계수</div>', unsafe_allow_html=True)
            r2_cols = st.columns(len(eval_df))
            for i, row in eval_df.iterrows():
                with r2_cols[i]:
                    r2_val = max(0, row['R²'])
                    fig_gauge = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=r2_val,
                        title={'text': row['Model'], 'font': {'size': 13}},
                        gauge={
                            'axis': {'range': [0, 1]},
                            'bar': {'color': '#1f77b4'},
                            'steps': [
                                {'range': [0, 0.5], 'color': '#ffcccc'},
                                {'range': [0.5, 0.8], 'color': '#ffe4b5'},
                                {'range': [0.8, 1.0], 'color': '#ccffcc'},
                            ],
                            'threshold': {'line': {'color': 'red', 'width': 2}, 'value': 0.8}
                        },
                        number={'valueformat': '.3f'}
                    ))
                    fig_gauge.update_layout(height=200, margin=dict(l=10,r=10,t=40,b=10))
                    st.plotly_chart(fig_gauge, use_container_width=True)

            # 잔차 분석
            st.markdown('<div class="section-title">📉 잔차(Residual) 분석</div>', unsafe_allow_html=True)
            selected_model_resid = st.selectbox("분석할 모델 선택", list(results.keys()))
            residuals = test['y'].values - results[selected_model_resid]['test_pred']

            col_res1, col_res2, col_res3 = st.columns(3)
            with col_res1:
                fig_res = go.Figure()
                fig_res.add_trace(go.Scatter(y=residuals, mode='lines+markers',
                                              line=dict(color='#d62728'), marker=dict(size=4)))
                fig_res.add_hline(y=0, line_dash='dash', line_color='black')
                fig_res.update_layout(title='잔차 시계열', height=250,
                                       margin=dict(l=0,r=0,t=40,b=0))
                st.plotly_chart(fig_res, use_container_width=True)

            with col_res2:
                fig_res_hist = px.histogram(x=residuals, nbins=20, title='잔차 분포',
                                             color_discrete_sequence=['#ff7f0e'])
                fig_res_hist.update_layout(height=250, margin=dict(l=0,r=0,t=40,b=0))
                st.plotly_chart(fig_res_hist, use_container_width=True)

            with col_res3:
                fig_scatter = go.Figure()
                fig_scatter.add_trace(go.Scatter(
                    x=results[selected_model_resid]['test_pred'], y=test['y'].values,
                    mode='markers', marker=dict(color='#2ca02c', size=5, opacity=0.7)
                ))
                min_val = min(min(test['y'].values), min(results[selected_model_resid]['test_pred']))
                max_val = max(max(test['y'].values), max(results[selected_model_resid]['test_pred']))
                fig_scatter.add_trace(go.Scatter(
                    x=[min_val, max_val], y=[min_val, max_val],
                    mode='lines', line=dict(color='red', dash='dash'), name='완벽한 예측'
                ))
                fig_scatter.update_layout(
                    title='실제값 vs 예측값', height=250,
                    xaxis_title='예측값', yaxis_title='실제값',
                    margin=dict(l=0,r=0,t=40,b=0)
                )
                st.plotly_chart(fig_scatter, use_container_width=True)

            # 전체 종합 인사이트
            st.markdown('<div class="section-title">💬 자동 분석 인사이트</div>', unsafe_allow_html=True)
            best_row = eval_df[eval_df['Model'] == best_model].iloc[0]
            adf_stationary = adf_test(ts['y'])['is_stationary']

            insights = []
            insights.append(f"✅ **최적 모델**: {best_model} (RMSE = {best_row['RMSE']:.4f})")
            insights.append(f"📌 **정상성**: 시계열은 {'정상(Stationary)' if adf_stationary else '비정상(Non-Stationary)'}입니다.")
            if best_row['MAPE (%)'] < 5:
                insights.append("🎯 MAPE < 5% → **매우 우수한** 예측 정확도")
            elif best_row['MAPE (%)'] < 10:
                insights.append("👍 MAPE < 10% → **양호한** 예측 정확도")
            elif best_row['MAPE (%)'] < 20:
                insights.append("⚠️ MAPE < 20% → **보통** 예측 정확도, 모델 개선 고려")
            else:
                insights.append("❌ MAPE ≥ 20% → **낮은** 예측 정확도, 모델 재검토 필요")
            if best_row['R²'] > 0.8:
                insights.append("📈 R² > 0.8 → 예측 모델이 **데이터 분산을 잘 설명**합니다.")

            for ins in insights:
                st.markdown(ins)
