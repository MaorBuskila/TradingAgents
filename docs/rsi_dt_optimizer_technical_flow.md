# RSI DT Optimizer — Full Technical Flow (Hybrid XGBoost & FinBERT Integration)

> **File:** `tradingagents/dataflows/rsi_dt_optimizer.py`  
> **Logic:** Hybrid XGBoost Classifier + RSI Mean Reversion + Sentiment & Regime Filtering

---

## 1. System Overview
The system upgrades a classic RSI Mean Reversion strategy by encapsulating it within a **Hybrid AI-Driven Architecture**[cite: 461]. It layers an XGBoost classifier, trend-following momentum indicators (EMA/MACD), Market Regime identification, and FinBERT sentiment analysis into a single decision-making unit[cite: 412]. The primary goal is to dynamically adapt to volatility and suspend trades during heavy bearish news cycles[cite: 462, 465].

## 2. Feature Engineering 
The predictive model uses 10 distinct features (standardized and scaled using `StandardScaler`) to enhance signal quality[cite: 534, 543]:
1. **RSI (14):** Detects core mean-reversion opportunities[cite: 530].
2. **EMA 50 & 200:** Captures short and long-term trend direction[cite: 529].
3. **EMA Ratio:** Non-linear interaction term calculated as EMA50 / EMA200[cite: 533].
4. **MACD & Signal Line:** Measures momentum via exponential moving average differentials[cite: 530].
5. **MACD Histogram:** Non-linear behavior encoding calculated as MACD - Signal[cite: 533].
6. **Bollinger Bands:** Detects volatility expansion and contraction[cite: 531].
7. **ATR & Volatility:** Uses rolling standard deviation for dynamic position sizing and risk management[cite: 532].
8. **FinBERT Sentiment Score ($S_t$):** Daily aggregated NLP score from financial news, normalized between -1 and 1[cite: 537].

## 3. Hybrid Logic & Risk Management
- **Market Regime Detection:** Calculates a 20-day Simple Moving Average (SMA) of daily percent changes[cite: 549]. The regime works as an on-off switch, constraining long trades strictly to Bullish environments ($R_t > 0$)[cite: 550].
- **ATR Position Sizing:** Replaces static percentage vetoes with dynamic risk caps[cite: 466]. Capital risk is set at 1% of total cash, and position sizing is dynamically allocated based on volatility, capped at a maximum of 10% of total cash per trade[cite: 554, 557].
- **Sentiment Shield (FinBERT):** Acts as a hard risk filter[cite: 537]. If the daily average sentiment score falls below **-0.70**, new trades are suspended and active trades are fully exited to defend against downside risk due to adverse news events[cite: 538].

## 4. Signal Fusion & Execution Logic
Signal generation relies on an additive hybrid scoring mechanism rather than isolated probabilities[cite: 557]:
- **Base Score:** The XGBoost model predicts the next-day return direction, yielding a `1` or `0`[cite: 540, 541]. This acts as the baseline score[cite: 557].
- **Trend Modifier:** If Price > EMA(50) AND MACD > Signal Line, `1` is added to the score[cite: 557].
- **RSI Modifier:** If RSI(14) < 30, `1` is added to the score[cite: 557].

**Entry Requirements:**
A **LONG** position is executed only if all of the following are true[cite: 558]:
1. The combined signal Score is $\ge 2$[cite: 558].
2. The Market Regime is Bullish[cite: 558].
3. The Price > EMA(200)[cite: 558].
4. There is no open position[cite: 558].

**Exit Conditions:**
The position is completely closed (Full Position Sell) if ANY of the following occur[cite: 558]:
1. The XGBoost model predicts a negative return (`0`)[cite: 558].
2. The Market Regime flips to Bearish[cite: 558].
3. RSI(14) reaches overbought levels ($> 70$)[cite: 558].
4. The Sentiment Score ($S_t$) drops below -0.70[cite: 558].