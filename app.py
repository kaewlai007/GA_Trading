import random
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

try:
    import yfinance as yf
except ImportError:
    yf = None

st.set_page_config(page_title="Trading Strategy Optimizer", page_icon="📈", layout="wide")
st.markdown("""
<style>
.block-container {padding-top: 1rem; padding-bottom: 1.2rem; max-width: 1550px;}
div[data-testid="stMetric"] {background: rgba(90,140,180,.08); border: 1px solid rgba(120,160,190,.2); padding: 12px; border-radius: 12px;}
.small-note {font-size: .86rem; color: #8a98a8;}
</style>
""", unsafe_allow_html=True)

st.title("📈 Trading Strategy Optimizer")
st.caption("Genetic Algorithm · Technical Indicators · Backtesting · Risk-adjusted Strategy Evaluation")

# ---------- Data helpers ----------
def load_price_data(uploaded, ticker, period):
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        df.columns = [str(c).strip() for c in df.columns]
        lower_map = {c.lower(): c for c in df.columns}
        date_col = next((lower_map[k] for k in ["date", "datetime", "timestamp"] if k in lower_map), None)
        close_col = next((lower_map[k] for k in ["close", "adj close", "adj_close", "price"] if k in lower_map), None)
        if close_col is None:
            raise ValueError("CSV ต้องมีคอลัมน์ Close (และควรมี Date ด้วย)")
        if date_col:
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
            df = df.dropna(subset=[date_col]).sort_values(date_col).set_index(date_col)
        else:
            df.index = pd.RangeIndex(start=0, stop=len(df), name="Row")
        df["Close"] = pd.to_numeric(df[close_col], errors="coerce")
        df = df.dropna(subset=["Close"])
        if "Volume" not in df.columns:
            df["Volume"] = 0
        return df[["Close", "Volume"]].astype({"Close": float, "Volume": float})
    if yf is None:
        raise ValueError("ยังไม่ได้ติดตั้ง yfinance ให้รัน: python -m pip install yfinance")
    df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError("ดาวน์โหลดข้อมูลไม่สำเร็จ ลองเปลี่ยน Ticker หรือ Period")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if "Close" not in df:
        raise ValueError("ไม่พบข้อมูล Close จากแหล่งข้อมูล")
    if "Volume" not in df:
        df["Volume"] = 0
    return df[["Close", "Volume"]].dropna()


def add_indicators(df):
    data = df.copy()
    data["Return"] = data["Close"].pct_change().fillna(0)
    data["RSI"] = 50.0
    delta = data["Close"].diff()
    gain = delta.clip(lower=0).rolling(14, min_periods=1).mean()
    loss = -delta.clip(upper=0).rolling(14, min_periods=1).mean()
    rs = gain / loss.replace(0, np.nan)
    data["RSI"] = (100 - (100 / (1 + rs))).fillna(50)
    return data



def evaluate_strategy(data, fast, slow, rsi_period, rsi_buy, rsi_sell, fee_pct,
                      indicator="SMA + RSI", direction="Long"):
    close = data["Close"]
    fast = int(fast)
    slow = int(slow)
    rsi_period = int(rsi_period)

    sma_fast = close.rolling(fast, min_periods=fast).mean()
    sma_slow = close.rolling(slow, min_periods=slow).mean()
    ema_fast = close.ewm(span=fast, adjust=False, min_periods=fast).mean()
    ema_slow = close.ewm(span=slow, adjust=False, min_periods=slow).mean()

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(rsi_period, min_periods=rsi_period).mean()
    loss = -delta.clip(upper=0).rolling(rsi_period, min_periods=rsi_period).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = (100 - 100 / (1 + rs)).fillna(50)

    macd_line = close.ewm(span=fast, adjust=False).mean() - close.ewm(span=slow, adjust=False).mean()
    macd_signal = macd_line.ewm(span=rsi_period, adjust=False).mean()
    bb_mid = close.rolling(slow, min_periods=slow).mean()
    bb_std = close.rolling(slow, min_periods=slow).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std

    # Build indicator-specific entry/exit signals.
    if indicator == "SMA Crossover":
        long_entry, long_exit = (sma_fast > sma_slow), (sma_fast < sma_slow)
        short_entry, short_exit = (sma_fast < sma_slow), (sma_fast > sma_slow)
    elif indicator == "EMA Crossover":
        long_entry, long_exit = (ema_fast > ema_slow), (ema_fast < ema_slow)
        short_entry, short_exit = (ema_fast < ema_slow), (ema_fast > ema_slow)
    elif indicator == "RSI":
        long_entry, long_exit = (rsi <= rsi_buy), (rsi >= rsi_sell)
        short_entry, short_exit = (rsi >= rsi_sell), (rsi <= rsi_buy)
    elif indicator == "MACD":
        long_entry, long_exit = (macd_line > macd_signal), (macd_line < macd_signal)
        short_entry, short_exit = (macd_line < macd_signal), (macd_line > macd_signal)
    elif indicator == "Bollinger Bands":
        long_entry, long_exit = (close < bb_lower), (close > bb_mid)
        short_entry, short_exit = (close > bb_upper), (close < bb_mid)
    else:  # combined SMA + RSI strategy (original behavior)
        long_entry = (sma_fast > sma_slow) & (rsi <= rsi_buy)
        long_exit = (sma_fast < sma_slow) | (rsi >= rsi_sell)
        short_entry = (sma_fast < sma_slow) & (rsi >= rsi_sell)
        short_exit = (sma_fast > sma_slow) | (rsi <= rsi_buy)

    position = pd.Series(0.0, index=data.index)
    holding = 0.0
    for i in range(len(data)):
        le = bool(long_entry.iloc[i]) if pd.notna(long_entry.iloc[i]) else False
        lx = bool(long_exit.iloc[i]) if pd.notna(long_exit.iloc[i]) else False
        se = bool(short_entry.iloc[i]) if pd.notna(short_entry.iloc[i]) else False
        sx = bool(short_exit.iloc[i]) if pd.notna(short_exit.iloc[i]) else False
        if holding == 0:
            if direction in ("Long", "Both") and le:
                holding = 1.0
            elif direction in ("Short", "Both") and se:
                holding = -1.0
        elif holding > 0 and (lx or (direction == "Short" and se)):
            holding = -1.0 if direction == "Both" and se else 0.0
        elif holding < 0 and (sx or (direction == "Long" and le)):
            holding = 1.0 if direction == "Both" and le else 0.0
        position.iloc[i] = holding

    # Execute signal on next bar to reduce look-ahead bias.
    position = position.shift(1).fillna(0)
    trades = position.diff().abs().fillna(position.abs())
    daily_returns = close.pct_change().fillna(0)
    strategy_returns = position * daily_returns - trades * (fee_pct / 100.0)
    equity = (1 + strategy_returns).cumprod()
    buy_hold = (1 + daily_returns).cumprod()

    total_return = float(equity.iloc[-1] - 1)
    buy_hold_return = float(buy_hold.iloc[-1] - 1)
    daily_std = float(strategy_returns.std(ddof=0))
    sharpe = float(strategy_returns.mean() / daily_std * np.sqrt(252)) if daily_std > 0 else 0.0
    drawdown = equity / equity.cummax() - 1
    max_drawdown = float(drawdown.min())
    fitness = total_return + 0.15 * sharpe + 0.5 * max_drawdown

    active_returns = strategy_returns[position != 0]
    win_rate = float((active_returns > 0).mean() * 100) if len(active_returns) else 0.0
    gross_profit = float(strategy_returns[strategy_returns > 0].sum())
    gross_loss = float(-strategy_returns[strategy_returns < 0].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0.0)
    metrics = {
        "Total Return": total_return, "Buy & Hold": buy_hold_return,
        "Sharpe Ratio": sharpe, "Max Drawdown": max_drawdown,
        "Trades": int((trades > 0).sum()), "Win Rate (active days)": win_rate,
        "Profit Factor (daily)": profit_factor,
    }
    curves = pd.DataFrame({
        "Strategy Equity": equity, "Buy & Hold Equity": buy_hold,
        "Drawdown": drawdown, "Position": position, "RSI": rsi,
        "Fast MA": sma_fast, "Slow MA": sma_slow,
        "Fast EMA": ema_fast, "Slow EMA": ema_slow,
        "MACD": macd_line, "MACD Signal": macd_signal,
        "BB Upper": bb_upper, "BB Middle": bb_mid, "BB Lower": bb_lower,
    })
    return fitness, metrics, curves


def run_ga(
    data, population_size, generations, crossover_rate,
    mutation_rate, elite_count, fee_pct, seed, indicator, direction
):
    rng = random.Random(seed)

    # Chromosome:
    # [Fast MA, Slow MA, RSI Period, RSI Buy, RSI Sell]
    def random_chromosome():
        fast = rng.randint(3, 30)
        slow = rng.randint(max(fast + 2, 10), 100)
        rsi_period = rng.randint(5, 30)
        rsi_buy = rng.randint(20, 45)
        rsi_sell = rng.randint(max(rsi_buy + 10, 55), 80)
        return [fast, slow, rsi_period, rsi_buy, rsi_sell]

    def repair(ch):
        fast = int(np.clip(round(ch[0]), 3, 30))
        slow = int(np.clip(round(ch[1]), max(fast + 2, 10), 100))
        rsi_period = int(np.clip(round(ch[2]), 5, 30))
        rsi_buy = int(np.clip(round(ch[3]), 20, 45))
        rsi_sell = int(np.clip(round(ch[4]), max(rsi_buy + 10, 55), 80))

        return [fast, slow, rsi_period, rsi_buy, rsi_sell]

    cache = {}

    def score(ch):
        key = tuple(ch)
        if key not in cache:
            cache[key] = evaluate_strategy(
                data,
                fast=ch[0],
                slow=ch[1],
                rsi_period=ch[2],
                rsi_buy=ch[3],
                rsi_sell=ch[4],
                fee_pct=fee_pct, indicator=indicator, direction=direction,
            )[0]
        return cache[key]

    population = [
        random_chromosome() for _ in range(population_size)
    ]

    history = []
    best = None
    best_score = -np.inf

    for gen in range(1, generations + 1):
        scored = sorted(
            [(score(ch), ch) for ch in population],
            key=lambda item: item[0],
            reverse=True,
        )

        if scored[0][0] > best_score:
            best_score = scored[0][0]
            best = scored[0][1].copy()

        history.append({
            "Generation": gen,
            "Best Fitness": float(scored[0][0]),
            "Average Fitness": float(
                np.mean([value for value, _ in scored])
            ),
        })

        # Elitism: preserve the best chromosomes
        new_population = [
            chromosome.copy()
            for _, chromosome in scored[:elite_count]
        ]

        def tournament():
            contestants = rng.sample(scored, min(3, len(scored)))
            return max(contestants, key=lambda item: item[0])[1]

        while len(new_population) < population_size:
            parent1 = tournament().copy()
            parent2 = tournament().copy()
            child1, child2 = parent1.copy(), parent2.copy()

            # Single-point crossover
            if rng.random() < crossover_rate:
                point = rng.randint(1, len(parent1) - 1)
                child1 = parent1[:point] + parent2[point:]
                child2 = parent2[:point] + parent1[point:]

            for child in (child1, child2):
                # Mutate individual genes
                if rng.random() < mutation_rate:
                    child[0] += rng.choice([-4, -2, 2, 4])
                if rng.random() < mutation_rate:
                    child[1] += rng.choice([-10, -5, 5, 10])
                if rng.random() < mutation_rate:
                    child[2] += rng.choice([-3, -1, 1, 3])
                if rng.random() < mutation_rate:
                    child[3] += rng.choice([-5, -2, 2, 5])
                if rng.random() < mutation_rate:
                    child[4] += rng.choice([-5, -2, 2, 5])

                new_population.append(repair(child))

                if len(new_population) >= population_size:
                    break

        population = new_population

    return best, pd.DataFrame(history), best_score

# ---------- Sidebar ----------
with st.sidebar:
    st.header("⚙️ Experiment Setup")
    source = st.radio("Price data source", ["Download from Yahoo Finance", "Upload CSV"], index=0)
    uploaded = None
    ticker, period = "AAPL", "5y"
    if source == "Download from Yahoo Finance":
        ticker = st.text_input("Ticker", value="AAPL", help="ตัวอย่าง AAPL, MSFT, TSLA, BTC-USD, ^GSPC")
        period = st.selectbox("Historical period", ["1y", "2y", "5y", "10y", "max"], index=2)
    else:
        uploaded = st.file_uploader("Upload price data (CSV)", type=["csv"])
    st.divider()
    st.subheader("📊 Trading Strategy")
    indicators = st.multiselect(
        "Select Indicator(s)",
        ["SMA + RSI", "SMA Crossover", "EMA Crossover", "RSI", "MACD", "Bollinger Bands"],
        default=["SMA + RSI"],
        help="เลือกหนึ่งตัวเพื่อทดลองเดี่ยว หรือหลายตัวเพื่อให้ GA ทดลองแต่ละ Indicator แยกกัน แล้วเปรียบเทียบคะแนน"
    )
    if not indicators:
        st.warning("กรุณาเลือกอย่างน้อยหนึ่ง Indicator")
        indicators = ["SMA + RSI"]
    direction = st.selectbox(
        "Trading Direction", ["Long", "Short", "Both"], index=0,
        help="Long = ซื้อแล้วถือ, Short = ขายชอร์ต, Both = ใช้ได้ทั้งสองฝั่ง"
    )
    st.caption(
        "GA ค้นหาพารามิเตอร์ใน chromosome 5 ค่า; แต่ละ Indicator ใช้เฉพาะพารามิเตอร์ที่เกี่ยวข้อง "
        "ค่าบางตัวจึงอาจไม่ส่งผลต่อสัญญาณของ Indicator ที่เลือก"
    )
    st.divider()
    st.subheader("🧬 Genetic Algorithm")
    population_size = st.slider("Population", 6, 40, 16, 2)
    generations = st.slider("Generations", 3, 40, 12)
    crossover_rate = st.slider("Crossover rate", 0.0, 1.0, 0.8, 0.05)
    mutation_rate = st.slider("Mutation rate", 0.0, 1.0, 0.15, 0.05)
    elite_count = st.slider("Elites", 1, 5, 2)
    seed = st.number_input("Random seed", min_value=0, max_value=999999, value=2026)
    st.divider()
    fee_pct = st.number_input("Trading fee per position change (%)", min_value=0.0, max_value=2.0, value=0.1, step=0.05)
    run_button = st.button("▶ Optimize Strategy", type="primary", use_container_width=True)

# ---------- Main ----------
if source == "Upload CSV" and uploaded is None:
    st.info("อัปโหลดไฟล์ CSV ที่มีคอลัมน์ Close (และ Date ถ้ามี) ทางแถบด้านซ้าย หรือเปลี่ยนเป็น Yahoo Finance เพื่อโหลดข้อมูลราคา")
    st.stop()
try:
    prices = load_price_data(uploaded if source == "Upload CSV" else None, ticker, period)
    if len(prices) < 120:
        st.warning("ข้อมูลราคามีน้อยกว่า 120 แถว ผลการ Optimize อาจไม่น่าเชื่อถือ")
    data = add_indicators(prices)
except Exception as exc:
    st.error(f"โหลดข้อมูลไม่สำเร็จ: {exc}")
    st.stop()

st.markdown("### 📅 Backtest Date Range")
if isinstance(data.index, pd.DatetimeIndex):
    min_date = data.index.min().date()
    max_date = data.index.max().date()
    date_cols = st.columns(2)
    start_date = date_cols[0].date_input("Start Date", value=min_date, min_value=min_date, max_value=max_date)
    end_date = date_cols[1].date_input("End Date", value=max_date, min_value=min_date, max_value=max_date)
    if start_date > end_date:
        st.error("Start Date ต้องไม่อยู่หลัง End Date")
        st.stop()
    data = data.loc[(data.index.date >= start_date) & (data.index.date <= end_date)].copy()
    if len(data) < 30:
        st.warning("ช่วงวันที่ที่เลือกมีข้อมูลน้อยกว่า 30 แถว ผล Backtest อาจไม่น่าเชื่อถือ")
else:
    st.caption("ไฟล์ CSV ไม่มีคอลัมน์วันที่ จึงใช้ลำดับแถวทั้งหมดเป็นช่วงข้อมูล")
    start_date, end_date = None, None

st.markdown("### 📁 Market Data Overview")
a, b, c, d = st.columns(4)
a.metric("Asset / Source", ticker if source == "Download from Yahoo Finance" else "Uploaded CSV")
b.metric("Price rows", f"{len(data):,}")
c.metric("First close", f"{data['Close'].iloc[0]:,.2f}")
d.metric("Latest close", f"{data['Close'].iloc[-1]:,.2f}")
with st.expander("Preview historical prices"):
    st.line_chart(data[["Close"]], height=220)
    st.dataframe(data.tail(10), use_container_width=True)

if "trading_result" not in st.session_state:
    st.session_state["trading_result"] = None

if run_button:
    if elite_count >= population_size:
        st.error("Elites ต้องน้อยกว่า Population")
    else:
        with st.spinner("Genetic Algorithm กำลังค้นหาและเปรียบเทียบกลยุทธ์..."):
            candidates = []
            comparisons = []
            for selected_indicator in indicators:
                candidate_best, candidate_history, candidate_fitness = run_ga(
                    data, int(population_size), int(generations), float(crossover_rate),
                    float(mutation_rate), int(elite_count), float(fee_pct), int(seed),
                    selected_indicator, direction
                )
                _, candidate_metrics, candidate_curves = evaluate_strategy(
                    data, candidate_best[0], candidate_best[1], candidate_best[2],
                    candidate_best[3], candidate_best[4], float(fee_pct),
                    indicator=selected_indicator, direction=direction
                )
                comparisons.append({
                    "Indicator": selected_indicator,
                    "Fitness": candidate_fitness,
                    "Total Return": candidate_metrics["Total Return"],
                    "Sharpe Ratio": candidate_metrics["Sharpe Ratio"],
                    "Max Drawdown": candidate_metrics["Max Drawdown"],
                    "Trades / Position Changes": candidate_metrics["Trades"],
                })
                candidates.append({
                    "best": candidate_best, "history": candidate_history,
                    "fitness": candidate_fitness, "metrics": candidate_metrics,
                    "curves": candidate_curves, "indicator": selected_indicator,
                })
            chosen = max(candidates, key=lambda item: item["fitness"])
            st.session_state["trading_result"] = {
                **chosen, "comparison": pd.DataFrame(comparisons),
                "ticker": ticker if source == "Download from Yahoo Finance" else "Uploaded CSV",
                "fee": float(fee_pct), "direction": direction,
                "indicators_tested": indicators.copy(),
                "start_date": str(start_date) if start_date else None,
                "end_date": str(end_date) if end_date else None,
            }

result = st.session_state.get("trading_result")
tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🧬 GA Optimization", "📉 Backtesting", "⚙️ Best Strategy"])
if result is None:
    with tab1:
        st.info("ตั้งค่าด้านซ้ายแล้วกด Optimize Strategy เพื่อเริ่มค้นหากลยุทธ์")
    with tab2:
        st.caption("GA จะใช้ Selection, Crossover, Mutation และ Elitism เพื่อค้นหาพารามิเตอร์")
    with tab3:
        st.caption("ผล Backtesting จะปรากฏหลัง Optimize")
    with tab4:
        st.caption("พารามิเตอร์กลยุทธ์ที่ดีที่สุดจะแสดงหลัง Optimize")
else:
    best = result["best"]
    metrics = result["metrics"]
    with tab1:
        st.subheader("Optimization Summary")
        st.caption(f"Best Indicator: {result.get('indicator', 'SMA + RSI')} · Direction: {result.get('direction', 'Long')}")
        st.caption("เมื่อเลือกหลาย Indicator ระบบจะ optimize แยกทีละตัว แล้วเลือกผลที่มี Fitness สูงสุด ไม่ได้รวมสัญญาณหลาย Indicator เป็นกลยุทธ์เดียว")
        if len(result.get("indicators_tested", [])) > 1:
            st.markdown("**Indicator Comparison**")
            st.dataframe(result["comparison"].sort_values("Fitness", ascending=False), use_container_width=True, hide_index=True)
        x1, x2, x3, x4 = st.columns(4)
        x1.metric("Strategy Return", f"{metrics['Total Return']:.2%}")
        x2.metric("Buy & Hold Return", f"{metrics['Buy & Hold']:.2%}")
        x3.metric("Sharpe Ratio", f"{metrics['Sharpe Ratio']:.2f}")
        x4.metric("Maximum Drawdown", f"{metrics['Max Drawdown']:.2%}")
        st.line_chart(result["curves"][["Strategy Equity", "Buy & Hold Equity"]], height=300)
        st.caption("ผลจำลองจากข้อมูลในอดีต ไม่ใช่การรับประกันผลตอบแทนในอนาคต")
    with tab2:
        st.subheader("Fitness Progress by Generation")
        st.line_chart(result["history"].set_index("Generation"), height=280)
        st.dataframe(result["history"], use_container_width=True, hide_index=True)
        st.markdown("**Best Chromosome**")
        st.code(
    f"[Fast MA={best[0]}, Slow MA={best[1]}, "
    f"RSI Period={best[2]}, RSI Buy={best[3]}, "
    f"RSI Sell={best[4]}]\nIndicator={result.get('indicator', 'SMA + RSI')}, Direction={result.get('direction', 'Long')}",
    language="text"
)
        st.caption("Chromosome = ชุดพารามิเตอร์ของกลยุทธ์; Fitness ให้คะแนนจากผลตอบแทน, Sharpe Ratio และการหักโทษ Maximum Drawdown")
    with tab3:
        st.subheader("Backtesting Performance")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Return", f"{metrics['Total Return']:.2%}")
        m2.metric("Trades / position changes", f"{metrics['Trades']}")
        m3.metric("Win Rate (active days)", f"{metrics['Win Rate (active days)']:.2f}%")
        m4.metric("Max Drawdown", f"{metrics['Max Drawdown']:.2%}")
        st.line_chart(result["curves"][["Strategy Equity", "Buy & Hold Equity"]], height=300)

        st.subheader("📍 Buy / Sell Signals")
        signal_frame = pd.DataFrame({"Close": data["Close"], "Position": result["curves"]["Position"]}, index=data.index)
        previous_position = signal_frame["Position"].shift(1).fillna(0)
        buy_mask = (signal_frame["Position"] > 0) & (previous_position <= 0)
        sell_mask = (signal_frame["Position"] <= 0) & (previous_position > 0)
        short_mask = (signal_frame["Position"] < 0) & (previous_position >= 0)
        cover_mask = (signal_frame["Position"] >= 0) & (previous_position < 0)
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(signal_frame.index, signal_frame["Close"], label="Close Price", linewidth=1.4)
        ax.scatter(signal_frame.index[buy_mask], signal_frame.loc[buy_mask, "Close"], marker="^", s=70, label="Buy / Long", zorder=3)
        ax.scatter(signal_frame.index[sell_mask], signal_frame.loc[sell_mask, "Close"], marker="v", s=70, label="Sell / Exit Long", zorder=3)
        ax.scatter(signal_frame.index[short_mask], signal_frame.loc[short_mask, "Close"], marker="v", s=70, label="Open Short", zorder=3)
        ax.scatter(signal_frame.index[cover_mask], signal_frame.loc[cover_mask, "Close"], marker="^", s=70, label="Cover Short", zorder=3)
        ax.set_title(f"{result.get('ticker', 'Asset')} — {result.get('indicator')} signals")
        ax.set_xlabel("Date / Row")
        ax.set_ylabel("Price")
        ax.grid(True, alpha=.25)
        ax.legend(loc="best", ncol=2)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        
        st.subheader("📊 Risk & Trade Statistics")

        r1, r2, r3 = st.columns(3)
        r1.metric("Win Rate (active days)", f"{metrics['Win Rate (active days)']:.2f}%")
        r2.metric(
            "Profit Factor (daily)",
            f"{metrics['Profit Factor (daily)']:.2f}"
            if np.isfinite(metrics["Profit Factor (daily)"])
            else "∞"
        )
        r3.metric("Number of Position Changes", f"{metrics['Trades']}")

        st.subheader("📈 Technical Indicators")
        if result.get("indicator") == "EMA Crossover":
            st.line_chart(result["curves"][["Fast EMA", "Slow EMA"]], height=220)
        elif result.get("indicator") == "MACD":
            st.line_chart(result["curves"][["MACD", "MACD Signal"]], height=220)
        elif result.get("indicator") == "Bollinger Bands":
            st.line_chart(result["curves"][["BB Upper", "BB Middle", "BB Lower"]], height=220)
        else:
            st.line_chart(result["curves"][["Fast MA", "Slow MA"]], height=220)
        st.line_chart(result["curves"][["RSI"]], height=180)

        with st.expander("ℹ️ How to read these metrics"):
            st.markdown("""
            - **Total Return:** ผลตอบแทนสะสมของกลยุทธ์หลังหัก Trading Fee ตามแบบจำลอง
            - **Win Rate (active days):** สัดส่วนวันที่ถือสถานะแล้วผลตอบแทนรายวันเป็นบวก ไม่ใช่ Win Rate ของเทรดที่ปิดแล้ว
            - **Profit Factor (daily):** คำนวณจากผลตอบแทนรายวันบวกเทียบกับผลตอบแทนรายวันลบ ไม่ใช่กำไรรวมต่อรอบเทรด
            - **Maximum Drawdown:** การลดลงสูงสุดของ Equity จากจุดสูงสุดก่อนหน้า
            - **Buy/Sell markers:** จุดเปลี่ยนสถานะที่ปรากฏบนกราฟราคาปิด
            """)
        st.subheader("⬇️ Export Results")

        backtest_export = result["curves"].copy()
        backtest_export["Indicator"] = result.get("indicator", "SMA + RSI")
        backtest_export["Direction"] = result.get("direction", "Long")
        st.download_button(
            "Download Backtest Results (CSV)",
            data=backtest_export.to_csv(index=True).encode("utf-8-sig"),
            file_name="trading_backtest_results.csv",
            mime="text/csv",
        )

        fitness_export = result["history"].copy()
        fitness_export["Indicator"] = result.get("indicator", "SMA + RSI")
        fitness_export["Direction"] = result.get("direction", "Long")
        st.download_button(
            "Download Fitness History (CSV)",
            data=fitness_export.to_csv(index=False).encode("utf-8-sig"),
            file_name="ga_fitness_history.csv",
            mime="text/csv",
        )

        strategy_params = pd.DataFrame([{
            "Indicator": result.get("indicator", "SMA + RSI"),
            "Direction": result.get("direction", "Long"),
            "Fast MA": best[0],
            "Slow MA": best[1],
            "RSI Period": best[2],
            "RSI Buy": best[3],
            "RSI Sell": best[4],
            "Fitness": result["fitness"],
            "Trading Fee (%)": result["fee"],
        }])

        st.download_button(
            "Download Best Strategy (CSV)",
            data=strategy_params.to_csv(index=False).encode("utf-8-sig"),
            file_name="best_trading_strategy.csv",
            mime="text/csv",
        )



        st.dataframe(pd.DataFrame({"Metric": list(metrics.keys()), "Value": list(metrics.values())}), use_container_width=True, hide_index=True)
    with tab4:
       
        st.subheader("Optimized Trading Rules")
        st.write(f"Indicator: **{result.get('indicator', 'SMA + RSI')}** · Direction: **{result.get('direction', 'Long')}**")
        p1, p2, p3, p4, p5 = st.columns(5)

        p1.metric("Fast MA", f"{best[0]} days")
        p2.metric("Slow MA", f"{best[1]} days")
        p3.metric("RSI Period", f"{best[2]} days")
        p4.metric("RSI Buy", f"{best[3]}")
        p5.metric("RSI Sell", f"{best[4]}")

        st.markdown("**Rule interpretation**")
        selected_indicator = result.get("indicator", "SMA + RSI")
        selected_direction = result.get("direction", "Long")

        if selected_indicator == "SMA + RSI":
            rule_text = (
                f"ใช้ SMA {best[0]} วัน และ SMA {best[1]} วันร่วมกับ RSI {best[2]} วัน; "
                f"ระดับ RSI เข้า = {best[3]} และออก = {best[4]}."
            )
        elif selected_indicator == "SMA Crossover":
            rule_text = f"ใช้ SMA crossover: Fast SMA {best[0]} วัน เทียบกับ Slow SMA {best[1]} วัน."
        elif selected_indicator == "EMA Crossover":
            rule_text = f"ใช้ EMA crossover: Fast EMA {best[0]} วัน เทียบกับ Slow EMA {best[1]} วัน."
        elif selected_indicator == "RSI":
            rule_text = f"ใช้ RSI {best[2]} วัน; ระดับเข้า = {best[3]} และระดับออก = {best[4]}."
        elif selected_indicator == "MACD":
            rule_text = (
                f"ใช้ MACD จาก EMA {best[0]} และ {best[1]} วัน "
                f"โดยใช้ EMA {best[2]} วันเป็น signal line."
            )
        else:
            rule_text = (
                f"ใช้ Bollinger Bands จากค่าเฉลี่ย {best[1]} วัน "
                "และแถบห่างจากค่าเฉลี่ย 2 standard deviations."
            )

        st.write(f"- **Indicator rule:** {rule_text}")
        direction_text = {
            "Long": "เปิดสถานะซื้อ (Long) เท่านั้น",
            "Short": "เปิดสถานะขายชอร์ต (Short) เท่านั้น",
            "Both": "อนุญาตทั้งสถานะ Long และ Short",
        }.get(selected_direction, selected_direction)
        st.write(f"- **Trading direction:** {direction_text}")
        st.write(f"- **Trading fee:** {result['fee']:.2f}% ต่อการเปลี่ยนสถานะหนึ่งหน่วย")

        if selected_indicator in ("SMA Crossover", "EMA Crossover"):
            st.caption("หมายเหตุ: RSI Period และ RSI thresholds ไม่ได้เป็นตัวกำหนดสัญญาณของ Crossover indicator นี้")
        elif selected_indicator == "RSI":
            st.caption("หมายเหตุ: Fast/Slow MA ไม่ได้เป็นตัวกำหนดสัญญาณของ RSI indicator นี้")
        elif selected_indicator == "Bollinger Bands":
            st.caption("หมายเหตุ: Fast MA, RSI Period และ RSI thresholds ไม่ได้เป็นตัวกำหนดสัญญาณ Bollinger Bands ในเวอร์ชันนี้")
        elif selected_indicator == "MACD":
            st.caption("หมายเหตุ: RSI Buy/Sell ไม่ได้เป็นตัวกำหนดสัญญาณ MACD ในเวอร์ชันนี้")
        st.warning(
            "นี่คือโปรโตไทป์เพื่อการศึกษา ผล Backtesting เป็นการจำลองจากข้อมูลในอดีตเท่านั้น "
            "และยังไม่รวม slippage, ภาษี, การส่งคำสั่งจริง "
            "หรือการป้องกัน overfitting"
        )


st.markdown("---")
st.caption("Trading Strategy Optimizer | Genetic Algorithm for strategy parameter search · Historical backtest only")
