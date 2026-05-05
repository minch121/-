# 📈 시계열 분석 & 예측 웹앱

산업데이터공학과 시계열분석 1차 프로젝트

---

## 🚀 빠른 시작

### 1. 로컬 실행

```bash
# 1. 패키지 설치
pip install -r requirements.txt

# 2. 앱 실행
streamlit run app.py
```

브라우저에서 `http://localhost:8501` 접속

---

## ☁️ Streamlit Community Cloud 배포 (무료)

### 단계별 배포 방법

**1단계: GitHub 저장소 생성**
1. https://github.com 에서 새 저장소 생성 (Public)
2. 이 폴더의 모든 파일 업로드

**2단계: Streamlit Cloud 배포**
1. https://share.streamlit.io 접속
2. GitHub 계정 연동
3. "New app" 클릭
4. 저장소, 브랜치(main), 파일(app.py) 선택
5. "Deploy!" 클릭

**3단계: 완료**
- 약 2~3분 후 `https://[앱이름].streamlit.app` 링크 생성

---

## 📁 프로젝트 구조

```
timeseries-webapp/
├── app.py                      # 메인 Streamlit 앱
├── requirements.txt            # 패키지 목록
├── README.md
├── modules/
│   ├── preprocessing.py        # 데이터 전처리
│   ├── analysis.py             # 정상성 검정, 분해, ACF/PACF
│   ├── forecasting.py          # 예측 모델 (SMA, ES, H-W, ARIMA, Prophet)
│   └── evaluation.py           # 평가지표
└── sample_data/
    ├── sample_daily.csv        # 일별 샘플 데이터
    └── sample_monthly.csv      # 월별 샘플 데이터
```

---

## 🔧 주요 기능

| 기능 | 설명 |
|---|---|
| CSV 자동 파싱 | 날짜/값 컬럼 자동 감지 |
| 결측치 처리 | 보간법, 전방/후방채움, 평균, 제거 선택 가능 |
| 정상성 검정 | ADF + KPSS 검정 |
| 시계열 분해 | 가법/승법 모델 선택 가능 |
| ACF/PACF | 자기상관 시각화 |
| 예측 모델 | SMA, 지수평활, Holt-Winters, ARIMA(자동), Prophet |
| 시평 조절 | 슬라이더로 7~365스텝 실시간 변경 |
| 평가 대시보드 | MAE, RMSE, MAPE, R², 잔차 분석 |
| 자동 인사이트 | 최적 모델 자동 추천 |

---

## 📋 CSV 형식

```csv
date,value
2024-01-01,120.5
2024-01-02,125.3
2024-01-03,118.7
```

- 첫 번째 컬럼: 날짜 (다양한 형식 자동 인식)
- 두 번째 컬럼: 수치형 값 (단변량)
