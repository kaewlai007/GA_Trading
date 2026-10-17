import streamlit as st
import yfinance as yf
import pandas as pd
st.set_page_config(
   page_title="GA Trading Optimizer",
   page_icon="📈",
   layout="wide"
)
st.title("📈 Genetic Algorithm Trading")
st.write("ระบบทดลองปรับกลยุทธ์ Trading ด้วย Genetic Algorithm")
st.divider()
st.header("1. ตั้งค่าเงินทุน")
capital = st.number_input(
   "เงินทุนเริ่มต้น (USD)",
   min_value=100,
   max_value=1000000,
   value=10000,
   step=1000
)
st.metric("เงินทุนเริ่มต้น", f"${capital:,.2f}")
st.divider()
st.header("2. ตั้งค่ากลยุทธ์ Trading")
col1, col2 = st.columns(2)
with col1:
   short_window = st.number_input(
       "Short Moving Average",
       min_value=2,
       max_value=50,
       value=5
   )
with col2:
   long_window = st.number_input(
       "Long Moving Average",
       min_value=10,
       max_value=200,
       value=20
   )
if short_window >= long_window:
   st.warning("Short MA ควรน้อยกว่า Long MA")
else:
   st.success(
       f"Short MA = {short_window}, Long MA = {long_window}"
   )
st.divider()
st.header("3. ข้อมูลราคาหุ้นย้อนหลัง")
symbol = st.text_input(
   "Ticker Symbol",
   value="AAPL",
   help="ตัวอย่าง AAPL = Apple, MSFT = Microsoft"
).strip().upper()
period = st.selectbox(
   "ช่วงเวลาข้อมูล",
   ["1mo", "3mo", "6mo", "1y", "2y"],
   index=3
)
if st.button("ดึงข้อมูลราคาหุ้น"):
   if not symbol:
       st.error("กรุณากรอก Ticker Symbol")
   else:
       with st.spinner("กำลังดึงข้อมูล..."):
           try:
               data = yf.download(
                   symbol,
                   period=period,
                   auto_adjust=True,
                   progress=False
               )
               if data.empty:
                   st.error("ไม่พบข้อมูล กรุณาตรวจสอบ Ticker Symbol")
               else:
                   if isinstance(data.columns, pd.MultiIndex):
                       data.columns = data.columns.get_level_values(0)
                   data = data.reset_index()
                   st.success(f"โหลดข้อมูล {symbol} สำเร็จ")
                   st.line_chart(
                       data.set_index("Date")["Close"],
                       y_label="Price (USD)"
                   )
                   st.subheader("ข้อมูลราคาล่าสุด")
                   st.dataframe(data.tail(10), use_container_width=True)
           except Exception as e:
               st.error(f"ดึงข้อมูลไม่สำเร็จ: {e}")
st.info("ตรวจสอบอินเทอร์เน็ตและ Ticker Symbol อีกครั้ง")
# 4. ระบบ Backtest กลยุทธ์ Trading
st.divider()
st.header("4. Backtest กลยุทธ์ Trading")
st.write(
   "จำลองการซื้อขายย้อนหลังด้วยกลยุทธ์ "
   "Moving Average Crossover"
)
if st.button("เริ่ม Backtest"):
   try:
       # ดาวน์โหลดข้อมูลหุ้น
       prices = yf.download(
           symbol,
           period=period,
           auto_adjust=True,
           progress=False
       )
       if prices.empty:
           st.error("ไม่พบข้อมูลราคาหุ้น")
       else:
           # รองรับข้อมูลคอลัมน์หลายระดับ
           if isinstance(prices.columns, pd.MultiIndex):
               prices.columns = prices.columns.get_level_values(0)
           prices = prices[["Close"]].copy()
           prices["Short_MA"] = prices["Close"].rolling(
               window=int(short_window)
           ).mean()
           prices["Long_MA"] = prices["Close"].rolling(
               window=int(long_window)
           ).mean()
           # สัญญาณ: ถือหุ้นเมื่อ Short MA > Long MA
           prices["Signal"] = (
               prices["Short_MA"] > prices["Long_MA"]
           ).astype(int)
           # ใช้สัญญาณจากวันก่อนหน้า เพื่อลด look-ahead bias
           prices["Position"] = prices["Signal"].shift(1).fillna(0)
           # ผลตอบแทนรายวันของหุ้นและกลยุทธ์
           prices["Market_Return"] = prices["Close"].pct_change().fillna(0)
           prices["Strategy_Return"] = (
               prices["Position"] * prices["Market_Return"]
           )
           # จำลองเงินทุนสะสม
           prices["Equity"] = capital * (
               1 + prices["Strategy_Return"]
           ).cumprod()
           # เปรียบเทียบกับการซื้อแล้วถือ
           prices["Buy_Hold_Equity"] = capital * (
               1 + prices["Market_Return"]
           ).cumprod()
           total_return = (
               prices["Equity"].iloc[-1] / capital - 1
           ) * 100
           buy_hold_return = (
               prices["Buy_Hold_Equity"].iloc[-1] / capital - 1
           ) * 100
           running_peak = prices["Equity"].cummax()
           drawdown = prices["Equity"] / running_peak - 1
           max_drawdown = drawdown.min() * 100
           col1, col2, col3 = st.columns(3)
           col1.metric("เงินทุนปลายงวด", f"${prices['Equity'].iloc[-1]:,.2f}")
           col2.metric("ผลตอบแทนกลยุทธ์", f"{total_return:.2f}%")
           col3.metric("Max Drawdown", f"{max_drawdown:.2f}%")
           st.subheader("กราฟเงินทุนจำลอง")
           st.line_chart(
               prices[["Equity", "Buy_Hold_Equity"]],
               y_label="Equity (USD)"
           )
           st.subheader("ข้อมูล Backtest ล่าสุด")
           st.dataframe(prices.tail(10), use_container_width=True)
           st.caption(
               f"Buy & Hold Return: {buy_hold_return:.2f}% "
               "| ผลจำลองยังไม่รวมค่าธรรมเนียมและ Slippage"
           )
   except Exception as e:
       st.error(f"Backtest ไม่สำเร็จ: {e}")
# 5. Genetic Algorithm Optimizer
import random
st.divider()
st.header("5. Genetic Algorithm Optimizer")
st.write(
   "ให้ Genetic Algorithm ทดลองค้นหาค่า Short MA "
   "และ Long MA ที่ให้ผล Backtest ดี"
)
population_size = st.slider(
   "Population Size",
   min_value=10,
   max_value=50,
   value=20,
   step=5
)
generations = st.slider(
   "จำนวน Generations",
   min_value=3,
   max_value=20,
   value=5
)
if st.button("เริ่ม Genetic Algorithm"):
   try:
       # โหลดข้อมูลราคา
       prices = yf.download(
           symbol,
           period=period,
           auto_adjust=True,
           progress=False
       )
       if prices.empty:
           st.error("ไม่พบข้อมูลราคาหุ้น")
       else:
           if isinstance(prices.columns, pd.MultiIndex):
               prices.columns = prices.columns.get_level_values(0)
           close = prices["Close"].dropna()
           if len(close) < 30:
               st.error("ข้อมูลน้อยเกินไปสำหรับการ Optimize")
           else:
               # Fitness: ผลตอบแทนสะสม หักค่าปรับจาก Max Drawdown
               def evaluate_strategy(short_ma, long_ma):
                   if short_ma >= long_ma:
                       return -1e9, None
                   short_avg = close.rolling(short_ma).mean()
                   long_avg = close.rolling(long_ma).mean()
                   signal = (short_avg > long_avg).astype(int)
                   position = signal.shift(1).fillna(0)
                   returns = close.pct_change().fillna(0)
                   strategy_returns = position * returns
                   equity = (1 + strategy_returns).cumprod()
                   total_return = equity.iloc[-1] - 1
                   drawdown = equity / equity.cummax() - 1
                   max_drawdown = abs(drawdown.min())
                   # คะแนนนี้เป็นเกณฑ์ทดลอง ไม่ใช่การรับประกันกำไร
                   fitness = total_return - 0.5 * max_drawdown
                   return fitness, {
                       "Short MA": short_ma,
                       "Long MA": long_ma,
                       "Return (%)": total_return * 100,
                       "Max Drawdown (%)": max_drawdown * 100,
                       "Fitness": fitness
                   }
               # Encoding + Generate: chromosome = [short, long]
               population = []
               while len(population) < population_size:
                   short_ma = random.randint(2, 20)
                   long_ma = random.randint(21, 100)
                   if short_ma < long_ma:
                       population.append([short_ma, long_ma])
               history = []
               progress = st.progress(0)
               status = st.empty()
               for generation in range(generations):
                   scored = []
                   for chromosome in population:
                       fitness, details = evaluate_strategy(
                           chromosome[0],
                           chromosome[1]
                       )
                       if details is not None:
                           scored.append((fitness, chromosome, details))
                   if not scored:
                       st.error("ไม่สามารถประเมินประชากรได้")
                       break
                   scored.sort(key=lambda item: item[0], reverse=True)
                   best_fitness, best_chromosome, best_details = scored[0]
                   history.append({
                       "Generation": generation + 1,
                       "Best Fitness": best_fitness,
                       "Best Short MA": best_chromosome[0],
                       "Best Long MA": best_chromosome[1]
                   })
                   status.write(
                       f"กำลังประเมิน Generation "
                       f"{generation + 1}/{generations}"
                   )
                   progress.progress((generation + 1) / generations)
                   # Selection: เลือกกลุ่มคะแนนสูงเป็นพ่อแม่
                   parent_pool = [
                       item[1] for item in scored[:max(2, len(scored) // 2)]
                   ]
                   new_population = [best_chromosome.copy()]
                   while len(new_population) < population_size:
                       parent1 = random.choice(parent_pool)
                       parent2 = random.choice(parent_pool)
                       # Crossover: ผสม Gene ของพ่อแม่
                       child = [
                           parent1[0],
                           parent2[1]
                       ]
                       # Mutation: สุ่มเปลี่ยน Gene บางตัว
                       if random.random() < 0.2:
                           child[0] = random.randint(2, 20)
                       if random.random() < 0.2:
                           child[1] = random.randint(21, 100)
                       if child[0] < child[1]:
                           new_population.append(child)
                   population = new_population
               if history:
                   st.success("Genetic Algorithm ทำงานเสร็จแล้ว")
                   final_scored = [
                       (evaluate_strategy(ch[0], ch[1])[0], ch)
                       for ch in population
                   ]
                   final_scored.sort(
                       key=lambda item: item[0],
                       reverse=True
                   )
                   # เลือกผลดีที่สุดจากประชากรรุ่นสุดท้าย
                   _, best = final_scored[0]
                   _, best_result = evaluate_strategy(best[0], best[1])
                   st.subheader("กลยุทธ์ที่ได้จากการ Optimize")
                   col1, col2, col3 = st.columns(3)
                   col1.metric("Short MA", best[0])
                   col2.metric("Long MA", best[1])
                   col3.metric(
                       "Return (%)",
                       f"{best_result['Return (%)']:.2f}%"
                   )
                   st.subheader("ผลลัพธ์แต่ละ Generation")
                   history_df = pd.DataFrame(history)
                   st.line_chart(
                       history_df.set_index("Generation")["Best Fitness"]
                   )
                   st.dataframe(history_df, use_container_width=True)
                   st.caption(
                       "ผลนี้เป็นการทดลองกับข้อมูลย้อนหลังชุดเดียว "
                       "ยังไม่ใช่หลักฐานว่ากลยุทธ์จะทำกำไรในอนาคต"
                   )
   except Exception as e:
       st.error(f"Genetic Algorithm ทำงานไม่สำเร็จ: {e}")
# 6. Initial Population Demo
st.divider()
st.header("6. Initial Population")
st.write(
   "สุ่มสร้างชุดค่า Moving Average "
   "เพื่อใช้เป็นประชากรเริ่มต้นของ Genetic Algorithm"
)
population_size_demo = st.slider(
   "จำนวน Chromosomes",
   min_value=5,
   max_value=20,
   value=10,
   key="population_size_demo"
)
if st.button("สร้าง Initial Population"):
   random.seed()
   initial_population = []
   for i in range(population_size_demo):
       short_ma = random.randint(2, 20)
       long_ma = random.randint(21, 100)
       initial_population.append({
           "Chromosome": i + 1,
           "Short MA": short_ma,
           "Long MA": long_ma
       })
   st.session_state["initial_population"] = initial_population
if "initial_population" in st.session_state:
   st.dataframe(
       st.session_state["initial_population"],
       use_container_width=True,
       hide_index=True
   )
   st.caption(
       "แต่ละ Chromosome แทนชุดพารามิเตอร์ "
       "Short MA และ Long MA ที่ GA สามารถนำไปประเมิน Fitness ได้"
   )
# 7. Fitness Evaluation
st.divider()
st.header("7. Fitness Evaluation")
st.write(
   "ประเมิน Chromosome แต่ละตัวจากผลตอบแทน "
   "ของกลยุทธ์ Moving Average บนข้อมูลหุ้น"
)
fitness_symbol = st.text_input(
   "Ticker สำหรับประเมิน Fitness",
   value="AAPL",
   key="fitness_symbol"
)
if st.button("ประเมิน Fitness"):
   try:
       fitness_data = yf.download(
           fitness_symbol,
           period="1y",
           auto_adjust=True,
           progress=False
       )
       if fitness_data.empty:
           st.error("ไม่พบข้อมูลหุ้น กรุณาตรวจสอบ Ticker")
       else:
           fitness_close = fitness_data["Close"]
           # รองรับข้อมูลที่มี MultiIndex
           if isinstance(fitness_close, pd.DataFrame):
               fitness_close = fitness_close.iloc[:, 0]
           fitness_close = fitness_close.dropna()
           population = st.session_state.get(
               "initial_population", []
           )
           if len(population) == 0:
               st.warning(
                   "กรุณากดสร้าง Initial Population "
                   "ในขั้นที่ 6 ก่อน"
               )
           else:
               fitness_results = []
               for chromosome in population:
                   short_ma = chromosome["Short MA"]
                   long_ma = chromosome["Long MA"]
                   short_line = fitness_close.rolling(
                       short_ma
                   ).mean()
                   long_line = fitness_close.rolling(
                       long_ma
                   ).mean()
                   signal = (short_line > long_line).astype(int)
                   position = signal.shift(1).fillna(0)
                   daily_returns = fitness_close.pct_change().fillna(0)
                   strategy_returns = position * daily_returns
                   equity = (1 + strategy_returns).cumprod()
                   total_return = (equity.iloc[-1] - 1) * 100
                   peak = equity.cummax()
                   drawdown = (equity / peak) - 1
                   max_drawdown = abs(drawdown.min()) * 100
                   # Fitness: ผลตอบแทนหักค่าปรับความเสี่ยง
                   fitness = total_return - (0.5 * max_drawdown)
                   fitness_results.append({
                       "Chromosome": chromosome["Chromosome"],
                       "Short MA": short_ma,
                       "Long MA": long_ma,
                       "Return (%)": round(total_return, 2),
                       "Max Drawdown (%)": round(max_drawdown, 2),
                       "Fitness": round(fitness, 2)
                   })
               results_df = pd.DataFrame(fitness_results)
               results_df = results_df.sort_values(
                   "Fitness", ascending=False
               ).reset_index(drop=True)
               st.session_state["fitness_results"] = results_df
               st.success("ประเมิน Fitness เรียบร้อยแล้ว")
               st.dataframe(
                   results_df,
                   use_container_width=True,
                   hide_index=True
               )
               best = results_df.iloc[0]
               st.subheader("Best Chromosome")
               st.write(f"Short MA: {int(best['Short MA'])}")
               st.write(f"Long MA: {int(best['Long MA'])}")
               st.write(f"Return: {best['Return (%)']}%")
               st.write(f"Max Drawdown: {best['Max Drawdown (%)']}%")
               st.write(f"Fitness: {best['Fitness']}")
   except Exception as e:
       st.error(f"เกิดข้อผิดพลาด: {e}")
# 8. Selection
st.divider()
st.header("8. Selection - การคัดเลือก")
st.write(
   "คัดเลือก Chromosomes ที่มี Fitness สูงที่สุด "
   "เพื่อใช้เป็นพ่อแม่ในกระบวนการ Crossover"
)
if st.button("เริ่ม Selection"):
   fitness_results = st.session_state.get("fitness_results")
   if fitness_results is None or fitness_results.empty:
       st.warning(
           "กรุณาสร้าง Initial Population "
           "และประเมิน Fitness ก่อน"
       )
   else:
       # เลือกครึ่งหนึ่งที่มี Fitness สูงสุด
       number_selected = max(2, len(fitness_results) // 2)
       selected = (
           fitness_results
           .sort_values("Fitness", ascending=False)
           .head(number_selected)
           .copy()
       )
       st.session_state["selected_population"] = selected
       st.success(
           f"Selection สำเร็จ: เลือก {len(selected)} Chromosomes"
       )
       st.dataframe(
           selected,
           use_container_width=True,
           hide_index=True
       )
       st.write("Chromosomes ที่ได้รับเลือกจะถูกนำไปใช้ในขั้น Crossover")
# 9. Crossover
st.divider()
st.header("9. Crossover - การผสมพันธุ์")
st.write(
   "นำยีนจากพ่อแม่ 2 ตัวมาผสมกัน "
   "เพื่อสร้าง Chromosome รุ่นลูก"
)
if st.button("เริ่ม Crossover"):
   selected = st.session_state.get("selected_population")
   if selected is None or len(selected) < 2:
       st.warning("กรุณาทำ Selection ให้สำเร็จก่อน")
   else:
       parents = selected.to_dict("records")
       offspring = []
       # สร้างลูกให้มีจำนวนเท่ากับพ่อแม่ที่ถูกเลือก
       for i in range(len(parents)):
           parent1 = random.choice(parents)
           parent2 = random.choice(parents)
           child_short = int(parent1["Short MA"])
           child_long = int(parent2["Long MA"])
           offspring.append({
               "Offspring": i + 1,
               "Parent 1 Short MA": int(parent1["Short MA"]),
               "Parent 1 Long MA": int(parent1["Long MA"]),
               "Parent 2 Short MA": int(parent2["Short MA"]),
               "Parent 2 Long MA": int(parent2["Long MA"]),
               "Child Short MA": child_short,
               "Child Long MA": child_long
           })
       offspring_df = pd.DataFrame(offspring)
       st.session_state["offspring_population"] = offspring_df
       st.success(f"Crossover สำเร็จ: สร้างลูก {len(offspring_df)} ตัว")
       st.dataframe(
           offspring_df,
           use_container_width=True,
           hide_index=True
       )
# 10. Mutation
st.divider()
st.header("10. Mutation - การกลายพันธุ์")
st.write(
   "สุ่มเปลี่ยนยีนบางค่าใน Offspring "
   "เพื่อเพิ่มความหลากหลายของประชากร"
)
mutation_rate = st.slider(
   "Mutation Rate",
   min_value=0.0,
   max_value=1.0,
   value=0.2,
   step=0.1,
   key="mutation_rate"
)
if st.button("เริ่ม Mutation"):
   offspring_df = st.session_state.get("offspring_population")
   if offspring_df is None or offspring_df.empty:
       st.warning("กรุณาทำ Crossover ให้สำเร็จก่อน")
   else:
       mutated = []
       for _, row in offspring_df.iterrows():
           old_short = int(row["Child Short MA"])
           old_long = int(row["Child Long MA"])
           new_short = old_short
           new_long = old_long
           short_changed = random.random() < mutation_rate
           long_changed = random.random() < mutation_rate
           if short_changed:
               new_short = random.randint(2, 20)
           if long_changed:
               new_long = random.randint(21, 100)
           mutated.append({
               "Offspring": int(row["Offspring"]),
               "Short MA Before": old_short,
               "Long MA Before": old_long,
               "Short MA After": new_short,
               "Long MA After": new_long,
               "Short Mutated": short_changed,
               "Long Mutated": long_changed
           })
       mutated_df = pd.DataFrame(mutated)
       st.session_state["mutated_population"] = mutated_df
       st.success("Mutation สำเร็จ")
       st.dataframe(
           mutated_df,
           use_container_width=True,
           hide_index=True
       )
       changed_count = sum(
           item["Short Mutated"] or item["Long Mutated"]
           for item in mutated
       )
       st.write(
           f"มี Offspring ที่ถูกเปลี่ยนอย่างน้อย 1 ยีน "
           f"{changed_count} จาก {len(mutated)} ตัว"
       )
# 11. New Generation Evaluation
st.divider()
st.header("11. New Generation Evaluation")
st.write(
   "ประเมิน Fitness ของ Offspring หลังผ่าน Crossover และ Mutation"
)
if st.button("ประเมินประชากรรุ่นใหม่"):
   mutated_df = st.session_state.get("mutated_population")
   if mutated_df is None or mutated_df.empty:
       st.warning("กรุณาทำ Mutation ให้สำเร็จก่อน")
   else:
       try:
           new_data = yf.download(
               fitness_symbol,
               period="1y",
               auto_adjust=True,
               progress=False
           )
           if new_data.empty:
               st.error("ไม่พบข้อมูลหุ้น กรุณาตรวจสอบ Ticker")
           else:
               new_close = new_data["Close"]
               if isinstance(new_close, pd.DataFrame):
                   new_close = new_close.iloc[:, 0]
               new_close = new_close.dropna()
               new_results = []
               for _, row in mutated_df.iterrows():
                   short_ma = int(row["Short MA After"])
                   long_ma = int(row["Long MA After"])
                   short_line = new_close.rolling(short_ma).mean()
                   long_line = new_close.rolling(long_ma).mean()
                   signal = (short_line > long_line).astype(int)
                   position = signal.shift(1).fillna(0)
                   daily_returns = new_close.pct_change().fillna(0)
                   strategy_returns = position * daily_returns
                   equity = (1 + strategy_returns).cumprod()
                   total_return = (equity.iloc[-1] - 1) * 100
                   peak = equity.cummax()
                   drawdown = equity / peak - 1
                   max_drawdown = abs(drawdown.min()) * 100
                   fitness = total_return - 0.5 * max_drawdown
                   new_results.append({
                       "Offspring": int(row["Offspring"]),
                       "Short MA": short_ma,
                       "Long MA": long_ma,
                       "Return (%)": round(total_return, 2),
                       "Max Drawdown (%)": round(max_drawdown, 2),
                       "Fitness": round(fitness, 2)
                   })
               new_df = pd.DataFrame(new_results).sort_values(
                   "Fitness", ascending=False
               ).reset_index(drop=True)
               st.session_state["new_generation_results"] = new_df
               st.success("ประเมินประชากรรุ่นใหม่เรียบร้อยแล้ว")
               st.dataframe(
                   new_df,
                   use_container_width=True,
                   hide_index=True
               )
               best_new = new_df.iloc[0]
               st.subheader("Best Offspring")
               st.write(f"Short MA: {int(best_new['Short MA'])}")
               st.write(f"Long MA: {int(best_new['Long MA'])}")
               st.write(f"Fitness: {best_new['Fitness']}")
       except Exception as e:
           st.error(f"เกิดข้อผิดพลาด: {e}")
# 12. Genetic Algorithm Iteration
st.divider()
st.header("12. Genetic Algorithm Iteration")
st.write(
   "ค้นหาค่า Moving Average ด้วย GA จากข้อมูลฝึก "
   "และประเมินผลกับข้อมูลทดสอบที่แยกไว้"
)
ga_generations = st.slider(
   "จำนวน Generations",
   min_value=3,
   max_value=20,
   value=5,
   key="iteration_generations"
)
if st.button("เริ่มรัน Genetic Algorithm"):
   try:
       ga_data = yf.download(
           fitness_symbol,
           period="2y",
           auto_adjust=True,
           progress=False
       )
       if ga_data.empty:
           st.error("ไม่พบข้อมูลหุ้น กรุณาตรวจสอบ Ticker")
       else:
           close = ga_data["Close"]
           if isinstance(close, pd.DataFrame):
               close = close.iloc[:, 0]
           close = close.dropna()
           if len(close) < 250:
               st.error(
                   "ข้อมูลราคาหุ้นไม่เพียงพอ "
                   "กรุณาเลือกข้อมูลย้อนหลังอย่างน้อยประมาณ 1 ปี"
               )
           else:
               # แบ่งข้อมูลตามเวลา: 80% ฝึก, 20% ทดสอบ
               split_index = int(len(close) * 0.8)
               train_close = close.iloc[:split_index]
               test_close = close.iloc[split_index:]
               def evaluate_strategy(prices, short_ma, long_ma):
                   short_line = prices.rolling(short_ma).mean()
                   long_line = prices.rolling(long_ma).mean()
                   signal = (short_line > long_line).astype(int)
                   position = signal.shift(1).fillna(0)
                   daily_returns = prices.pct_change().fillna(0)
                   strategy_returns = position * daily_returns
                   equity = (1 + strategy_returns).cumprod()
                   total_return = (equity.iloc[-1] - 1) * 100
                   drawdown = equity / equity.cummax() - 1
                   max_drawdown = abs(drawdown.min()) * 100
                   fitness = total_return - 0.5 * max_drawdown
                   return fitness, total_return, max_drawdown
               # สร้างประชากรเริ่มต้น
               population = [
                   [
                       random.randint(2, 20),
                       random.randint(21, 100)
                   ]
                   for _ in range(20)
               ]
               history = []
               best_overall = None
               for generation in range(1, ga_generations + 1):
                   scored_population = []
                   for chromosome in population:
                       fitness, ret, dd = evaluate_strategy(
                           train_close,
                           chromosome[0],
                           chromosome[1]
                       )
                       scored_population.append({
                           "Short MA": chromosome[0],
                           "Long MA": chromosome[1],
                           "Fitness": fitness,
                           "Return (%)": ret,
                           "Max Drawdown (%)": dd
                       })
                   scored_population.sort(
                       key=lambda item: item["Fitness"],
                       reverse=True
                   )
                   best = scored_population[0]
                   history.append({
                       "Generation": generation,
                       "Best Short MA": best["Short MA"],
                       "Best Long MA": best["Long MA"],
                       "Best Fitness": round(best["Fitness"], 2),
                       "Train Return (%)": round(
                           best["Return (%)"], 2
                       ),
                       "Train Max Drawdown (%)": round(
                           best["Max Drawdown (%)"], 2
                       )
                   })
                   if (
                       best_overall is None
                       or best["Fitness"] > best_overall["Fitness"]
                   ):
                       best_overall = best.copy()
                   # Selection: เลือกครึ่งบนของประชากร
                   parents = scored_population[:10]
                   # Elitism: เก็บตัวที่ดีที่สุดไว้
                   new_population = [[
                       best["Short MA"],
                       best["Long MA"]
                   ]]
                   # Crossover + Mutation
                   while len(new_population) < 20:
                       parent1 = random.choice(parents)
                       parent2 = random.choice(parents)
                       child_short = parent1["Short MA"]
                       child_long = parent2["Long MA"]
                       if random.random() < 0.2:
                           child_short = random.randint(2, 20)
                       if random.random() < 0.2:
                           child_long = random.randint(21, 100)
                       new_population.append([
                           child_short,
                           child_long
                       ])
                   population = new_population
               # ประเมิน Best Solution บนข้อมูลทดสอบ
               test_fitness, test_return, test_drawdown = (
                   evaluate_strategy(
                       test_close,
                       best_overall["Short MA"],
                       best_overall["Long MA"]
                   )
               )
               history_df = pd.DataFrame(history)
               st.session_state["ga_iteration_history"] = history_df
               st.session_state["ga_best_solution"] = {
                   "Short MA": best_overall["Short MA"],
                   "Long MA": best_overall["Long MA"],
                   "Train Fitness": best_overall["Fitness"],
                   "Test Fitness": test_fitness,
                   "Test Return (%)": test_return,
                   "Test Max Drawdown (%)": test_drawdown
               }
               st.success(
                   "Genetic Algorithm ทำงานครบทุก Generation แล้ว"
               )
               st.subheader("Best Fitness per Generation")
               st.line_chart(
                   history_df.set_index("Generation")["Best Fitness"]
               )
               st.subheader("Generation History")
               st.dataframe(
                   history_df,
                   use_container_width=True,
                   hide_index=True
               )
               st.subheader("Best Solution Found")
               col1, col2 = st.columns(2)
               col1.metric(
                   "Best Short MA",
                   best_overall["Short MA"]
               )
               col2.metric(
                   "Best Long MA",
                   best_overall["Long MA"]
               )
               st.write(
                   f"Training Fitness: "
                   f"{best_overall['Fitness']:.2f}"
               )
               st.subheader("ผลบนข้อมูลทดสอบ")
               col1, col2, col3 = st.columns(3)
               col1.metric(
                   "Test Return",
                   f"{test_return:.2f}%"
               )
               col2.metric(
                   "Test Max Drawdown",
                   f"{test_drawdown:.2f}%"
               )
               col3.metric(
                   "Test Fitness",
                   f"{test_fitness:.2f}"
               )
               st.caption(
                   "ข้อมูล 80% แรกใช้ค้นหาพารามิเตอร์ "
                   "และข้อมูล 20% ท้ายใช้ประเมินผลเบื้องต้น "
                   "การทดสอบนี้ยังไม่รวมค่าธรรมเนียมและ Slippage"
               )
   except Exception as e:
       st.error(f"เกิดข้อผิดพลาด: {e}")
# 13. Results Dashboard
st.divider()
st.header("13. Results - สรุปผลการทดลอง")
history = st.session_state.get("ga_iteration_history")
best_solution = st.session_state.get("ga_best_solution")
if history is None or history.empty or best_solution is None:
  st.info(
       "กรุณารัน Genetic Algorithm ในขั้นที่ 12 "
       "เพื่อสร้างผลการทดลองก่อน"
   )
else:
   col1, col2, col3 = st.columns(3)
   col1.metric(
       "Best Short MA",
       best_solution["Short MA"]
   )
   col2.metric(
       "Best Long MA",
       best_solution["Long MA"]
   )
   col3.metric(
       "Training Fitness",
       f"{best_solution['Train Fitness']:.2f}"
   )
   st.subheader("Generation History")
   st.dataframe(
       history,
       use_container_width=True,
       hide_index=True
   )
   st.subheader("Best Fitness per Generation")
   st.line_chart(
       history.set_index("Generation")["Best Fitness"]
   )
   st.subheader("ผลการทดสอบกับข้อมูลที่แยกไว้")
   col1, col2, col3 = st.columns(3)
   col1.metric(
       "Test Return",
       f"{best_solution['Test Return (%)']:.2f}%"
   )
   col2.metric(
       "Test Max Drawdown",
       f"{best_solution['Test Max Drawdown (%)']:.2f}%"
   )
   col3.metric(
       "Test Fitness",
       f"{best_solution['Test Fitness']:.2f}"
   )
   st.write(
       f"พารามิเตอร์ที่ GA เลือก: "
       f"Short MA = {best_solution['Short MA']}, "
       f"Long MA = {best_solution['Long MA']}"
   )
   st.caption(
       "Training Fitness ใช้สำหรับค้นหาพารามิเตอร์ "
       "ส่วน Test Return และ Test Fitness เป็นผลจากข้อมูลทดสอบ "
       "ผลย้อนหลังไม่รับประกันผลตอบแทนในอนาคต"
   )

# 14. Strategy Comparison
st.divider()
st.header("14. Strategy Comparison")
st.write(
   "เปรียบเทียบกลยุทธ์ Moving Average ที่ GA เลือก "
   "กับ Buy & Hold โดยใช้ข้อมูลทดสอบช่วงเดียวกัน"
)
comparison_symbol = st.text_input(
   "Ticker สำหรับเปรียบเทียบ",
   value="AAPL",
   key="comparison_symbol"
)
if st.button("เปรียบเทียบกลยุทธ์"):
   try:
       best_solution = st.session_state.get("ga_best_solution")
       if best_solution is None:
           st.warning(
               "กรุณารัน Genetic Algorithm ในขั้นที่ 12 ก่อน"
           )
       else:
           comparison_data = yf.download(
               comparison_symbol,
               period="2y",
               auto_adjust=True,
               progress=False
           )
           if comparison_data.empty:
               st.error(
                   "ไม่พบข้อมูลหุ้น กรุณาตรวจสอบ Ticker"
               )
           else:
               close = comparison_data["Close"]
               if isinstance(close, pd.DataFrame):
                   close = close.iloc[:, 0]
               close = close.dropna()
               if len(close) < 250:
                   st.error(
                       "ข้อมูลราคาหุ้นไม่เพียงพอสำหรับการเปรียบเทียบ"
                   )
               else:
                   # แบ่งช่วงข้อมูลเหมือนกับขั้นที่ 12
                   split_index = int(len(close) * 0.8)
                   test_start = close.index[split_index]
                   short_ma = int(best_solution["Short MA"])
                   long_ma = int(best_solution["Long MA"])
                   short_line = close.rolling(short_ma).mean()
                   long_line = close.rolling(long_ma).mean()
                   signal = (short_line > long_line).astype(int)
                   position = signal.shift(1).fillna(0)
                   daily_returns = close.pct_change().fillna(0)
                   # ใช้ผลตอบแทนเฉพาะช่วงทดสอบ
                   test_returns = daily_returns.loc[test_start:]
                   test_position = position.loc[test_start:]
                   strategy_returns = (
                       test_position * test_returns
                   )
                   strategy_equity = (
                       1 + strategy_returns
                   ).cumprod()
                   buy_hold_equity = (
                       1 + test_returns
                   ).cumprod()
                   strategy_return = (
                       strategy_equity.iloc[-1] - 1
                   ) * 100
                   buy_hold_return = (
                       buy_hold_equity.iloc[-1] - 1
                   ) * 100
                   comparison_df = pd.DataFrame({
                       "GA Moving Average": strategy_equity,
                       "Buy & Hold": buy_hold_equity
                   })
                   st.subheader(
                       "Cumulative Performance - Test Period"
                   )
                   st.line_chart(comparison_df)
                   col1, col2 = st.columns(2)
                   col1.metric(
                       "GA Strategy Return",
                       f"{strategy_return:.2f}%"
                   )
                   col2.metric(
                       "Buy & Hold Return",
                       f"{buy_hold_return:.2f}%"
                   )
                   st.write(
                       f"Short MA = {short_ma} | "
                       f"Long MA = {long_ma}"
                   )
                   st.caption(
                       "เปรียบเทียบในช่วงข้อมูลทดสอบเดียวกัน "
                       "โดยใช้พารามิเตอร์ที่ GA เลือกจากข้อมูลฝึก "
                       "ผลยังไม่รวมค่าธรรมเนียมและ Slippage "
                       "และไม่รับประกันผลตอบแทนในอนาคต"
                   )
   except Exception as e:
       st.error(f"เกิดข้อผิดพลาด: {e}")