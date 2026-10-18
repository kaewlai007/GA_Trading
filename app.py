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



def evaluate_strategy(data, fast, slow, rsi_period, rsi_buy, rsi_sell, fee_pct):
    close = data["Close"]

    fast_ma = close.rolling(int(fast), min_periods=int(fast)).mean()
    slow_ma = close.rolling(int(slow), min_periods=int(slow)).mean()

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(
        int(rsi_period), min_periods=int(rsi_period)
    ).mean()
    loss = -delta.clip(upper=0).rolling(
        int(rsi_period), min_periods=int(rsi_period)
    ).mean()

    rs = gain / loss.replace(0, np.nan)
    rsi = (100 - 100 / (1 + rs)).fillna(50)

    # Buy: bullish MA trend + RSI below buy threshold
    # Sell: bearish MA trend OR RSI above sell threshold
    entry = (fast_ma > slow_ma) & (rsi <= rsi_buy)
    exit_signal = (fast_ma < slow_ma) | (rsi >= rsi_sell)

    # Simulate Long/Cash positions
    position = pd.Series(0.0, index=data.index)
    holding = 0.0

    for i in range(len(data)):
        if holding == 0 and bool(entry.iloc[i]):
            holding = 1.0
        elif holding == 1 and bool(exit_signal.iloc[i]):
            holding = 0.0
        position.iloc[i] = holding

    # Execute the signal on the next bar to reduce look-ahead bias
    position = position.shift(1).fillna(0)

    trades = position.diff().abs().fillna(position.abs())
    daily_returns = data["Close"].pct_change().fillna(0)

    strategy_returns = (
        position * daily_returns - trades * (fee_pct / 100.0)
    )
    equity = (1 + strategy_returns).cumprod()
    buy_hold = (1 + daily_returns).cumprod()

    total_return = float(equity.iloc[-1] - 1)
    buy_hold_return = float(buy_hold.iloc[-1] - 1)

    daily_std = float(strategy_returns.std(ddof=0))
    sharpe = (
        float(strategy_returns.mean() / daily_std * np.sqrt(252))
        if daily_std > 0 else 0.0
    )

    drawdown = equity / equity.cummax() - 1
    max_drawdown = float(drawdown.min())

    # Fitness rewards return and risk-adjusted performance
    fitness = total_return + 0.15 * sharpe + 0.5 * max_drawdown

    # Approximate daily-return win rate and profit factor
    active_returns = strategy_returns[position > 0]
    if len(active_returns):
        win_rate = float((active_returns > 0).mean() * 100)
    else:
        win_rate = 0.0

    gross_profit = float(strategy_returns[strategy_returns > 0].sum())
    gross_loss = float(-strategy_returns[strategy_returns < 0].sum())
    profit_factor = (
        gross_profit / gross_loss if gross_loss > 0
        else (float("inf") if gross_profit > 0 else 0.0)
    )

    metrics = {
        "Total Return": total_return,
        "Buy & Hold": buy_hold_return,
        "Sharpe Ratio": sharpe,
        "Max Drawdown": max_drawdown,
        "Trades": int((trades > 0).sum()),
        "Win Rate (active days)": win_rate,
        "Profit Factor (daily)": profit_factor,
    }

    curves = pd.DataFrame({
        "Strategy Equity": equity,
        "Buy & Hold Equity": buy_hold,
        "Drawdown": drawdown,
        "Position": position,
        "RSI": rsi,
        "Fast MA": fast_ma,
        "Slow MA": slow_ma,
    })

    return fitness, metrics, curves



def run_ga(
    data, population_size, generations, crossover_rate,
    mutation_rate, elite_count, fee_pct, seed
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
                fee_pct=fee_pct,
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
        with st.spinner("Genetic Algorithm กำลังค้นหาค่ากลยุทธ์ที่เหมาะสม..."):
            best, history, fitness = run_ga(data, int(population_size), int(generations), float(crossover_rate), float(mutation_rate), int(elite_count), float(fee_pct), int(seed))
            
            _, metrics, curves = evaluate_strategy(
                data,
                best[0], best[1], best[2], best[3], best[4],
                float(fee_pct)
            )

            st.session_state["trading_result"] = {"best": best, "history": history, "fitness": fitness, "metrics": metrics, "curves": curves, "ticker": ticker if source == "Download from Yahoo Finance" else "Uploaded CSV", "fee": float(fee_pct)}

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
    f"RSI Sell={best[4]}]",
    language="text"
)
        st.caption("Chromosome = ชุดพารามิเตอร์ของกลยุทธ์; Fitness ให้คะแนนจากผลตอบแทน, Sharpe Ratio และการหักโทษ Maximum Drawdown")
    with tab3:
        st.subheader("Backtesting Performance")
        m1, m2, m3 = st.columns(3)
        m1.metric("Total Return", f"{metrics['Total Return']:.2%}")
        m2.metric("Trades / position changes", f"{metrics['Trades']}")
        m3.metric("Max Drawdown", f"{metrics['Max Drawdown']:.2%}")
        st.line_chart(result["curves"][["Strategy Equity", "Buy & Hold Equity"]], height=300)
        
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
        st.line_chart(result["curves"][["Fast MA", "Slow MA"]], height=220)
        st.line_chart(result["curves"][["RSI"]], height=180)

        st.subheader("⬇️ Export Results")

        st.download_button(
            "Download Backtest Results (CSV)",
            data=result["curves"].to_csv(index=True).encode("utf-8-sig"),
            file_name="trading_backtest_results.csv",
            mime="text/csv",
        )

        st.download_button(
            "Download Fitness History (CSV)",
            data=result["history"].to_csv(index=False).encode("utf-8-sig"),
            file_name="ga_fitness_history.csv",
            mime="text/csv",
        )

        strategy_params = pd.DataFrame([{
            "Fast MA": best[0],
            "Slow MA": best[1],
            "RSI Period": best[2],
            "RSI Buy": best[3],
            "RSI Sell": best[4],
            "Fitness": result["fitness"],
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
        p1, p2, p3, p4, p5 = st.columns(5)

        p1.metric("Fast MA", f"{best[0]} days")
        p2.metric("Slow MA", f"{best[1]} days")
        p3.metric("RSI Period", f"{best[2]} days")
        p4.metric("RSI Buy", f"{best[3]}")
        p5.metric("RSI Sell", f"{best[4]}")

        st.markdown("**Rule interpretation**")
        st.write(
            f"- เปิดสถานะ Long เมื่อ Fast MA ({best[0]}) "
            f"สูงกว่า Slow MA ({best[1]}) และ RSI ({best[2]} วัน) "
            f"ต่ำกว่าหรือเท่ากับ {best[3]}."
        )
        st.write(
            f"- ปิดสถานะเมื่อ Fast MA ต่ำกว่า Slow MA "
            f"หรือ RSI สูงกว่าหรือเท่ากับ {best[4]}."
        )
        st.write(
            f"- คิดค่าธรรมเนียม {result['fee']:.2f}% "
            f"ต่อการเปลี่ยนสถานะหนึ่งครั้ง"
        )
        st.warning(
            "นี่คือโปรโตไทป์เพื่อการศึกษา กลยุทธ์นี้เป็น Long/Cash เท่านั้น "
            "และยังไม่รวม slippage, ภาษี, การส่งคำสั่งจริง "
            "หรือการป้องกัน overfitting"
        )


st.markdown("---")
st.caption("Trading Strategy Optimizer | Genetic Algorithm for strategy parameter search · Historical backtest only")
