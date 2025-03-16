import streamlit as st
import requests
import pandas as pd
import datetime
import numpy as np
import xgboost as xgb
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from textblob import TextBlob  # 감성 분석 라이브러리
import ta  # 기술적 지표 계산 라이브러리

# Finnhub API Key (secrets.toml에서 가져오기)
API_KEY = st.secrets["FINNHUB_API_KEY"]

st.set_page_config(page_title="📈 뉴스 & 재무 분석 기반 주식 예측", layout="wide")
st.title("📰 뉴스 및 재무 데이터 반영 주식 매수/매도 추천")

# 📌 Finnhub 종목 검색 API 호출 함수
def search_stock(query):
    url = f"https://finnhub.io/api/v1/search?q={query}&token={API_KEY}"
    response = requests.get(url).json()
    return response.get("result", [])

# 📌 과거 주가 데이터 조회 함수
def get_historical_data(symbol):
    end_time = int(datetime.datetime.now().timestamp())  # 현재 시간
    start_time = end_time - (60 * 60 * 24 * 365)  # 365일 (1년) 전 데이터 조회로 확장

    url = f"https://finnhub.io/api/v1/stock/candle?symbol={symbol}&resolution=D&from={start_time}&to={end_time}&token={API_KEY}"
    response = requests.get(url).json()

    if "t" in response:
        df = pd.DataFrame({
            "Date": pd.to_datetime(response["t"], unit="s"),
            "Open": response["o"],
            "High": response["h"],
            "Low": response["l"],
            "Close": response["c"],
            "Volume": response["v"]
        })
        return df
    return None

# 📌 뉴스 데이터 조회 함수
def get_news(symbol):
    url = f"https://finnhub.io/api/v1/company-news?symbol={symbol}&from={datetime.date.today() - datetime.timedelta(days=14)}&to={datetime.date.today()}&token={API_KEY}"
    response = requests.get(url).json()
    return response if isinstance(response, list) else []

# 📌 재무 데이터 조회 함수
def get_financials(symbol):
    # 재무제표 데이터 가져오기
    income_url = f"https://finnhub.io/api/v1/stock/financials?symbol={symbol}&statement=ic&freq=annual&token={API_KEY}"
    balance_url = f"https://finnhub.io/api/v1/stock/financials?symbol={symbol}&statement=bs&freq=annual&token={API_KEY}"
    
    income_response = requests.get(income_url).json()
    balance_response = requests.get(balance_url).json()
    
    return {
        "income": income_response.get("financials", []),
        "balance": balance_response.get("financials", [])
    }

# 📌 기업 정보 조회 함수
def get_company_profile(symbol):
    url = f"https://finnhub.io/api/v1/stock/profile2?symbol={symbol}&token={API_KEY}"
    response = requests.get(url).json()
    return response

# 📌 뉴스 감성 분석 수행
def analyze_sentiment(news):
    sentiment_scores = []
    for article in news:
        title = article.get("headline", "")
        text = article.get("summary", "")
        sentiment = TextBlob(title + " " + text).sentiment.polarity  # 감성 점수 (-1~1)
        sentiment_scores.append({
            "headline": title,
            "date": datetime.datetime.fromtimestamp(article.get("datetime", 0)),
            "url": article.get("url", ""),
            "sentiment": sentiment
        })
    
    # 감성 점수 기준으로 정렬
    sentiment_scores.sort(key=lambda x: x["date"], reverse=True)
    
    # 평균 감성 점수 계산
    avg_sentiment = np.mean([item["sentiment"] for item in sentiment_scores]) if sentiment_scores else 0
    
    return sentiment_scores, avg_sentiment

# 📌 기술적 지표 계산 함수
def calculate_technical_indicators(df):
    # 이동평균선
    df["SMA_5"] = ta.trend.sma_indicator(df["Close"], window=5)
    df["SMA_20"] = ta.trend.sma_indicator(df["Close"], window=20)
    df["SMA_50"] = ta.trend.sma_indicator(df["Close"], window=50)
    df["SMA_200"] = ta.trend.sma_indicator(df["Close"], window=200)
    
    # RSI (Relative Strength Index)
    df["RSI"] = ta.momentum.rsi(df["Close"], window=14)
    
    # MACD (Moving Average Convergence Divergence)
    macd = ta.trend.MACD(df["Close"])
    df["MACD"] = macd.macd()
    df["MACD_Signal"] = macd.macd_signal()
    df["MACD_Hist"] = macd.macd_diff()
    
    # 볼린저 밴드
    bollinger = ta.volatility.BollingerBands(df["Close"])
    df["BB_High"] = bollinger.bollinger_hband()
    df["BB_Low"] = bollinger.bollinger_lband()
    df["BB_Mid"] = bollinger.bollinger_mavg()
    
    # ATR (Average True Range) - 변동성 지표
    df["ATR"] = ta.volatility.average_true_range(df["High"], df["Low"], df["Close"])
    
    return df

# 📌 예측 시각화 함수
def plot_stock_prediction(df, prediction, sentiment_score):
    fig = make_subplots(rows=3, cols=1, 
                        shared_xaxes=True, 
                        vertical_spacing=0.05, 
                        row_heights=[0.6, 0.2, 0.2],
                        subplot_titles=("가격 차트", "거래량", "기술적 지표 (RSI)"))
    
    # 캔들스틱 차트
    fig.add_trace(
        go.Candlestick(
            x=df['Date'],
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name="주가"
        ),
        row=1, col=1
    )
    
    # 이동평균선 추가
    fig.add_trace(go.Scatter(x=df['Date'], y=df['SMA_20'], name="20일 이동평균선", line=dict(color='blue')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['Date'], y=df['SMA_50'], name="50일 이동평균선", line=dict(color='red')), row=1, col=1)
    
    # 볼린저 밴드 추가
    fig.add_trace(go.Scatter(x=df['Date'], y=df['BB_High'], name="볼린저 상단", line=dict(color='rgba(0,128,0,0.3)')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['Date'], y=df['BB_Low'], name="볼린저 하단", line=dict(color='rgba(0,128,0,0.3)')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['Date'], y=df['BB_Mid'], name="볼린저 중앙", line=dict(color='rgba(0,128,0,0.8)')), row=1, col=1)
    
    # 거래량 차트
    fig.add_trace(
        go.Bar(
            x=df['Date'],
            y=df['Volume'],
            name="거래량",
            marker=dict(color='rgba(0,0,128,0.5)')
        ),
        row=2, col=1
    )
    
    # RSI 차트
    fig.add_trace(
        go.Scatter(
            x=df['Date'],
            y=df['RSI'],
            name="RSI",
            line=dict(color='purple')
        ),
        row=3, col=1
    )
    
    # RSI 기준선 추가 (30, 70)
    fig.add_trace(go.Scatter(x=df['Date'], y=[30] * len(df), name="RSI 30", line=dict(color='red', dash='dash')), row=3, col=1)
    fig.add_trace(go.Scatter(x=df['Date'], y=[70] * len(df), name="RSI 70", line=dict(color='red', dash='dash')), row=3, col=1)
    
    # 최근 10개 데이터 포인트만 표시
    last_date = df['Date'].iloc[-1]
    first_date = df['Date'].iloc[-60]  # 최근 60일 데이터만 표시
    
    # 예측 결과 추가
    last_date_str = last_date.strftime('%Y-%m-%d')
    prediction_text = f"예측: {'📈 상승' if prediction == 1 else '📉 하락'}"
    sentiment_text = f"뉴스 감성: {'긍정적 😊' if sentiment_score > 0 else '부정적 😞' if sentiment_score < 0 else '중립적 😐'}"
    
    # 차트 레이아웃 설정
    fig.update_layout(
        title=f"주가 분석 차트 ({last_date_str}) - {prediction_text}, {sentiment_text}",
        xaxis_rangeslider_visible=False,
        height=800,
        width=1000,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
    )
    
    # X축 범위 설정 (최근 데이터만 표시)
    fig.update_xaxes(range=[first_date, last_date], row=1, col=1)
    fig.update_xaxes(range=[first_date, last_date], row=2, col=1)
    fig.update_xaxes(range=[first_date, last_date], row=3, col=1)
    
    return fig

# 사이드바 설정
st.sidebar.title("📊 주식 분석 설정")
analysis_period = st.sidebar.slider("분석 기간 (일)", min_value=30, max_value=365, value=180, step=30)
news_period = st.sidebar.slider("뉴스 분석 기간 (일)", min_value=7, max_value=30, value=14, step=1)

# 🔍 종목 검색
query = st.text_input("🔍 조회할 주식명을 입력하세요 (예: Apple, Tesla, Samsung)").strip()

if query:
    results = search_stock(query)
    if results:
        stock_list = {f"{item['description']} ({item['symbol']})": item["symbol"] for item in results}
        selected_stock = st.selectbox("📌 조회할 종목을 선택하세요", list(stock_list.keys()))
        symbol = stock_list[selected_stock]
        
        # 기업 정보 가져오기
        company_profile = get_company_profile(symbol)
        
        # 기업 정보 표시
        if company_profile:
            col1, col2, col3 = st.columns([1, 2, 1])
            
            with col1:
                if "logo" in company_profile and company_profile["logo"]:
                    st.image(company_profile["logo"], width=100)
                    
            with col2:
                st.subheader(company_profile.get("name", symbol))
                st.write(f"**산업:** {company_profile.get('finnhubIndustry', '정보 없음')}")
                st.write(f"**웹사이트:** [{company_profile.get('weburl', '정보 없음')}]({company_profile.get('weburl', '#')})")
                
            with col3:
                st.metric(label="현재가", value=f"${company_profile.get('marketCapitalization', 0):,.2f}B")
        
        # 로딩 상태 표시
        with st.spinner("데이터를 불러오는 중..."):
            # 주가 데이터, 뉴스, 재무 데이터 가져오기
            df = get_historical_data(symbol)
            news = get_news(symbol)
            financials = get_financials(symbol)
            
            # 뉴스 감성 분석
            news_sentiment, avg_sentiment = analyze_sentiment(news)
            
            if df is not None:
                # 기술적 지표 계산
                df = calculate_technical_indicators(df)
                
                # 탭 생성
                tab1, tab2, tab3, tab4 = st.tabs(["📊 주가 예측", "📰 뉴스 분석", "💹 재무 분석", "🔍 기술적 지표"])
                
                with tab1:
                    # 머신러닝 학습 데이터 준비
                    df.dropna(inplace=True)
                    df["Target"] = np.where(df["Close"].shift(-1) > df["Close"], 1, 0)
                    
                    # 특성(Feature) 선택
                    features = ["Open", "High", "Low", "Close", "Volume", 
                              "SMA_5", "SMA_20", "SMA_50", "RSI", 
                              "MACD", "MACD_Signal", "ATR"]
                    
                    X = df[features]
                    y = df["Target"]
                    
                    # 학습/테스트 분할
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                    
                    # XGBoost 모델 생성 및 학습
                    model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, use_label_encoder=False, eval_metric="logloss")
                    model.fit(X_train, y_train)
                    
                    # 교차 검증
                    cv_scores = cross_val_score(model, X, y, cv=5)
                    
                    # 테스트 셋 예측
                    y_pred = model.predict(X_test)
                    accuracy = accuracy_score(y_test, y_pred)
                    
                    # 혼동 행렬 계산
                    cm = confusion_matrix(y_test, y_pred)
                    
                    # 최신 데이터로 예측
                    last_row = X.iloc[-1:].values
                    prediction = model.predict(last_row)[0]
                    pred_proba = model.predict_proba(last_row)[0]
                    
                    # 특성 중요도
                    importance = model.feature_importances_
                    feature_importance = pd.DataFrame({
                        'Feature': features,
                        'Importance': importance
                    }).sort_values(by='Importance', ascending=False)
                    
                    # 예측 결과 표시
                    st.subheader("📊 예측 결과")
                    
                    # 결과 지표 표시
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric(label="모델 정확도", value=f"{accuracy * 100:.2f}%")
                    with col2:
                        st.metric(label="교차 검증 정확도", value=f"{np.mean(cv_scores) * 100:.2f}%")
                    with col3:
                        st.metric(label="뉴스 감성 점수", value=f"{avg_sentiment:.3f}")
                    
                    # 예측 시각화
                    st.write("### 주가 및 예측 차트")
                    fig = plot_stock_prediction(df.iloc[-60:], prediction, avg_sentiment)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # 예측 신뢰도
                    st.write("### 예측 신뢰도")
                    pred_confidence = max(pred_proba) * 100
                    st.progress(float(pred_confidence) / 100)
                    st.write(f"모델 예측 신뢰도: {pred_confidence:.2f}%")
                    
                    # 특성 중요도 차트
                    st.write("### 특성 중요도")
                    st.bar_chart(data=feature_importance.set_index('Feature'))
                    
                    # 최종 매수/매도 추천
                    st.subheader("📌 최종 투자 추천")
                    
                    # 신호 강도 계산 (예측 + 감성 분석)
                    signal_strength = (pred_proba[1] - 0.5) * 2  # -1 ~ 1 사이 값으로 정규화
                    sentiment_factor = avg_sentiment  # 이미 -1 ~ 1 사이 값
                    
                    # 가중치 적용
                    combined_signal = (signal_strength * 0.7) + (sentiment_factor * 0.3)
                    
                    if combined_signal > 0.3:
                        st.success(f"📢 **강한 매수 추천** (신호 강도: {combined_signal:.2f})")
                        st.write("모델과 뉴스 분석 모두 강한 상승 가능성을 시사합니다.")
                    elif combined_signal > 0.1:
                        st.success(f"📢 **약한 매수 추천** (신호 강도: {combined_signal:.2f})")
                        st.write("상승 가능성이 있으나, 리스크 관리에 주의하세요.")
                    elif combined_signal < -0.3:
                        st.error(f"📢 **강한 매도 추천** (신호 강도: {combined_signal:.2f})")
                        st.write("모델과 뉴스 분석 모두 강한 하락 가능성을 시사합니다.")
                    elif combined_signal < -0.1:
                        st.error(f"📢 **약한 매도 추천** (신호 강도: {combined_signal:.2f})")
                        st.write("하락 가능성이 있으나, 추가 분석이 필요합니다.")
                    else:
                        st.info(f"⚖️ **관망 추천** (신호 강도: {combined_signal:.2f})")
                        st.write("현재 뚜렷한 매수/매도 신호가 없습니다. 추가 정보를 기다리거나 포지션을 유지하세요.")
                
                with tab2:
                    st.subheader("📰 뉴스 분석")
                    
                    # 전체 뉴스 감성 분포 시각화
                    if news_sentiment:
                        sentiment_values = [item["sentiment"] for item in news_sentiment]
                        
                        st.write("### 뉴스 감성 분포")
                        hist_data = pd.DataFrame({
                            "sentiment": sentiment_values
                        })
                        st.bar_chart(hist_data.sentiment.value_counts(bins=10).sort_index())
                        
                        # 평균 감성 점수 표시
                        if avg_sentiment > 0.2:
                            st.success(f"✅ 최근 뉴스 감성이 매우 긍정적입니다. (평균 점수: {avg_sentiment:.2f})")
                        elif avg_sentiment > 0:
                            st.success(f"✅ 최근 뉴스 감성이 약간 긍정적입니다. (평균 점수: {avg_sentiment:.2f})")
                        elif avg_sentiment < -0.2:
                            st.error(f"❌ 최근 뉴스 감성이 매우 부정적입니다. (평균 점수: {avg_sentiment:.2f})")
                        elif avg_sentiment < 0:
                            st.error(f"❌ 최근 뉴스 감성이 약간 부정적입니다. (평균 점수: {avg_sentiment:.2f})")
                        else:
                            st.info(f"⚖️ 최근 뉴스 감성이 중립적입니다. (평균 점수: {avg_sentiment:.2f})")
                        
                        # 뉴스 목록 표시
                        st.write("### 최근 뉴스 목록")
                        for i, item in enumerate(news_sentiment[:10]):  # 최근 10개 뉴스만 표시
                            sentiment_color = "green" if item["sentiment"] > 0 else "red" if item["sentiment"] < 0 else "gray"
                            date_str = item["date"].strftime("%Y-%m-%d")
                            
                            with st.expander(f"{date_str} - {item['headline'][:60]}..." if len(item['headline']) > 60 else f"{date_str} - {item['headline']}"):
                                st.write(f"**날짜:** {date_str}")
                                st.write(f"**제목:** {item['headline']}")
                                st.write(f"**감성 점수:** <span style='color:{sentiment_color}'>{item['sentiment']:.3f}</span>", unsafe_allow_html=True)
                                st.write(f"**링크:** [뉴스 보기]({item['url']})")
                    else:
                        st.warning("⚠️ 최근 뉴스 데이터가 없습니다.")
                
                with tab3:
                    st.subheader("💹 재무 분석")
                    
                    if financials["income"] or financials["balance"]:
                        # 재무제표 데이터 처리
                        if financials["income"]:
                            income_data = financials["income"][0]
                            
                            # 주요 지표 추출
                            revenue = income_data.get("revenue", 0)
                            ebitda = income_data.get("ebitda", 0)
                            net_income = income_data.get("netIncome", 0)
                            eps = income_data.get("eps", 0)
                            
                            # 재무 지표 표시
                            st.write("### 주요 재무 지표")
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.metric(label="매출 (Revenue)", value=f"${revenue:,}")
                                st.metric(label="EBITDA", value=f"${ebitda:,}")
                            
                            with col2:
                                st.metric(label="순이익 (Net Income)", value=f"${net_income:,}")
                                st.metric(label="주당순이익 (EPS)", value=f"${eps:,.2f}")
                            
                            # 수익성 지표 계산
                            if revenue > 0:
                                profit_margin = (net_income / revenue) * 100
                                ebitda_margin = (ebitda / revenue) * 100
                                
                                st.write("### 수익성 지표")
                                col3, col4 = st.columns(2)
                                
                                with col3:
                                    st.metric(label="순이익률 (Profit Margin)", value=f"{profit_margin:.2f}%")
                                
                                with col4:
                                    st.metric(label="EBITDA 마진", value=f"{ebitda_margin:.2f}%")
                    else:
                        st.warning("⚠️ 재무 데이터를 가져올 수 없습니다.")
                
                with tab4:
                    st.subheader("🔍 기술적 지표 분석")
                    
                    # 최근 데이터 표시
                    last_row = df.iloc[-1]
                    last_date = last_row.name if isinstance(last_row.name, pd.Timestamp) else df['Date'].iloc[-1]
                    
                    # 주요 기술적 지표 표시
                    st.write(f"### {last_date.strftime('%Y-%m-%d')} 기준 기술적 지표")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        # RSI 상태
                        rsi_value = last_row["RSI"]
                        if rsi_value > 70:
                            st.metric(label="RSI (14)", value=f"{rsi_value:.2f}", delta="과매수")
                        elif rsi_value < 30:
                            st.metric(label="RSI (14)", value=f"{rsi_value:.2f}", delta="과매도")
                        else:
                            st.metric(label="RSI (14)", value=f"{rsi_value:.2f}")
                    
                    with col2:
                        # MACD 상태
                        macd_value = last_row["MACD"]
                        macd_signal = last_row["MACD_Signal"]
                        macd_hist = last_row["MACD_Hist"]
                        
                        if macd_hist > 0:
                            delta_text = "상승 신호" if macd_hist > 0 else "상승 약화"
                            st.metric(label="MACD", value=f"{macd_value:.2f}", delta=delta_text)
                        else:
                            delta_text = "하락 신호" if macd_hist < 0 else "하락 약화"
                            st.metric(label="MACD", value=f"{macd_value:.2f}", delta=delta_text)
                    
                    with col3:
                        # 볼린저 밴드 상태
                        close_price = last_row["Close"]
                        bb_high = last_row["BB_High"]
                        bb_low = last_row["BB_Low"]
                        
                        if close_price > bb_high:
                            st.metric(label="볼린저 밴드", value=f"{close_price:.2f}", delta="상단 돌파")
                        elif close_price < bb_low:
                            st.metric(label="볼린저 밴드", value=f"{close_price:.2f}", delta="하단 돌파")
                        else:
                            percent_b = (close_price - bb_low) / (bb_high - bb_low)
                            st.metric(label="볼린저 밴드 %B", value=f"{percent_b:.2f}")
                    
                    # 이동평균선 크로스 확인
                    st.write("### 이동평균선 크로스 분석")
                    
                    ma_cross_signals = []
                    
                    # 골든 크로스 (단기>장기)
                    if df["SMA_5"].iloc[-1] > df["SMA_20"].iloc[-1] and df["SMA_5"].iloc[-2] <= df["SMA_20"].iloc[-2]:
                        ma_cross_signals.append("📈 단기 골든 크로스 발생 (5일선 > 20일선)")
                    
                    if df["SMA_20"].iloc[-1] > df["SMA_50"].iloc[-1] and df["SMA_20"].iloc[-2] <= df["SMA_50"].iloc[-2]:
                        ma_cross_signals.append("📈 중기 골든 크로스 발생 (20일선 > 50일선)")
                    
                    # 데드 크로스 (단기<장기)
                    if df["SMA_5"].iloc[-1] < df["SMA_20"].iloc[-1] and df["SMA_5"].iloc[-2] >= df["SMA_20"].iloc[-2]:
                        ma_cross_signals.append("📉 단기 데드 크로스 발생 (5일선 < 20일선)")
                    
                    if df["SMA_20"].iloc[-1] < df["SMA_50"].iloc[-1] and df["SMA_20"].iloc[-2] >= df["SMA_50"].iloc[-2]:
                        ma_cross_signals.append("📉 중기 데드 크로스 발생 (20일선 < 50일선)")
                    
                    # 이동평균선 기울기 확인
                    ma_slope_5 = (df["SMA_5"].iloc[-1] - df["SMA_5"].iloc[-5]) / df["SMA_5"].iloc[-5] * 100
                    ma_slope_20 = (df["SMA_20"].iloc[-1] - df["SMA_20"].iloc[-5]) / df["SMA_20"].iloc[-5] * 100
                    
                    if ma_slope_5 > 0:
                        ma_cross_signals.append(f"📈 5일 이동평균선 상승 중 (+{ma_slope_5:.2f}%)")
                    else:
                        ma_cross_signals.append(f"📉 5일 이동평균선 하락 중 ({ma_slope_5:.2f}%)")
                        
                    if ma_slope_20 > 0:
                        ma_cross_signals.append(f"📈 20일 이동평균선 상승 중 (+{ma_slope_20:.2f}%)")
                    else:
                        ma_cross_signals.append(f"📉 20일 이동평균선 하락 중 ({ma_slope_20:.2f}%)")
                    
                    # 시그널 표시
                    for signal in ma_cross_signals:
                        if "📈" in signal:
                            st.success(signal)
                        else:
                            st.error(signal)
                    
                    # 추가 기술적 지표 차트
                    st.write("### MACD & RSI 차트")
                    
                    # MACD 차트
                    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, subplot_titles=("MACD", "RSI"))
                    
                    # MACD 라인
                    fig.add_trace(
                        go.Scatter(x=df['Date'][-60:], y=df['MACD'][-60:], name="MACD", line=dict(color='blue')),
                        row=1, col=1
                    )
                    # 시그널 라인
                    fig.add_trace(
                        go.Scatter(x=df['Date'][-60:], y=df['MACD_Signal'][-60:], name="Signal", line=dict(color='red')),
                        row=1, col=1
                    )
                    # 히스토그램
                    fig.add_trace(
                        go.Bar(x=df['Date'][-60:], y=df['MACD_Hist'][-60:], name="Histogram", marker=dict(color='purple')),
                        row=1, col=1
                    )
                    
                    # RSI 차트
                    fig.add_trace(
                        go.Scatter(x=df['Date'][-60:], y=df['RSI'][-60:], name="RSI", line=dict(color='green')),
                        row=2, col=1
                    )
                    # 과매수/과매도 기준선
                    fig.add_trace(
                        go.Scatter(x=df['Date'][-60:], y=[70] * 60, name="과매수", line=dict(color='red', dash='dash')),
                        row=2, col=1
                    )
                    fig.add_trace(
                        go.Scatter(x=df['Date'][-60:], y=[30] * 60, name="과매도", line=dict(color='green', dash='dash')),
                        row=2, col=1
                    )
                    
                    fig.update_layout(height=600, showlegend=True)
                    st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("⚠️ 검색 결과가 없습니다. 올바른 종목명을 입력하세요.")
        symbol = None
else:
    st.info("👆 주식명을 입력하고 분석을 시작하세요.")
    st.write("""
    ### 📌 주요 기능 안내
    
    이 앱은 다음과 같은 기능을 제공합니다:
    
    1. **기술적 지표 분석**: RSI, MACD, 볼린저 밴드 등 다양한 기술적 지표를 통해 주가 추세를 분석합니다.
    2. **뉴스 감성 분석**: 최근 뉴스의 감성을 AI로 분석하여 시장 분위기를 파악합니다.
    3. **재무 데이터 분석**: 매출, 순이익 등 주요 재무 지표를 확인합니다.
    4. **머신러닝 예측**: 과거 데이터와 기술적 지표를 바탕으로 내일의 주가 방향을 예측합니다.
    5. **종합 투자 추천**: 모든 분석 결과를 종합하여 매수/매도 추천을 제공합니다.
    
    📊 **참고**: 이 앱의 예측 결과는 투자 참고 자료로만 활용하시고, 최종 투자 결정은 본인의 판단에 따라 신중하게 결정하세요.
    """)