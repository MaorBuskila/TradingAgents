from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    get_indicators,
    get_language_instruction,
    get_stock_data,
)
from tradingagents.dataflows.config import get_config


def create_market_analyst(llm):

    def market_analyst_node(state):
        current_date = state["trade_date"]
        instrument_context = build_instrument_context(state["company_of_interest"])

        # ── Adaptive RSI context from walk-forward optimization ─────────────
        rsi_period     = state.get("rsi_optimal_period", 14)
        rsi_upper      = state.get("rsi_optimal_upper",  70.0)
        rsi_lower      = state.get("rsi_optimal_lower",  30.0)
        rsi_oos_sharpe = state.get("rsi_oos_sharpe",     0.0)
        rsi_confidence = state.get("rsi_confidence",     "UNKNOWN")
        rsi_regime     = state.get("rsi_regime",         "unknown")
        rsi_from_cache = state.get("rsi_from_cache",     False)
        cache_label    = " (from cache)" if rsi_from_cache else " (freshly optimized)"

        adaptive_rsi_context = f"""
⚡ ADAPTIVE RSI OPTIMIZATION RESULTS{cache_label}:
  • Optimal Period    : {rsi_period}  (default RSI uses 14)
  • Overbought Level  : {rsi_upper}  (default 70)
  • Oversold Level    : {rsi_lower}  (default 30)
  • OOS Sharpe Ratio  : {rsi_oos_sharpe:.3f}
  • Confidence        : {rsi_confidence}
  • Market Regime     : {rsi_regime}

IMPORTANT: When selecting and interpreting RSI indicators, use RSI({rsi_period}) as your primary momentum indicator.
Apply {rsi_upper} as the overbought threshold and {rsi_lower} as the oversold threshold — NOT the standard 70/30 defaults.
These parameters were walk-forward validated on out-of-sample data for this specific ticker.
"""
        # ───────────────────────────────────────────────────────────────────

        tools = [
            get_stock_data,
            get_indicators,
        ]

        system_message = (
            adaptive_rsi_context
            + """You are a trading assistant tasked with analyzing financial markets. Your role is to select the **most relevant indicators** for a given market condition or trading strategy from the following list. The goal is to choose up to **8 indicators** that provide complementary insights without redundancy. Categories and each category's indicators are:

Moving Averages:
- close_50_sma: 50 SMA: A medium-term trend indicator. Usage: Identify trend direction and serve as dynamic support/resistance. Tips: It lags price; combine with faster indicators for timely signals.
- close_200_sma: 200 SMA: A long-term trend benchmark. Usage: Confirm overall market trend and identify golden/death cross setups. Tips: It reacts slowly; best for strategic trend confirmation rather than frequent trading entries.
- close_10_ema: 10 EMA: A responsive short-term average. Usage: Capture quick shifts in momentum and potential entry points. Tips: Prone to noise in choppy markets; use alongside longer averages for filtering false signals.

MACD Related:
- macd: MACD: Computes momentum via differences of EMAs. Usage: Look for crossovers and divergence as signals of trend changes. Tips: Confirm with other indicators in low-volatility or sideways markets.
- macds: MACD Signal: An EMA smoothing of the MACD line. Usage: Use crossovers with the MACD line to trigger trades. Tips: Should be part of a broader strategy to avoid false positives.
- macdh: MACD Histogram: Shows the gap between the MACD line and its signal. Usage: Visualize momentum strength and spot divergence early. Tips: Can be volatile; complement with additional filters in fast-moving markets.

Momentum Indicators:
- rsi: RSI (14-period default): Measures momentum to flag overbought/oversold conditions. Usage: Apply 70/30 thresholds and watch for divergence to signal reversals. Tips: In strong trends, RSI may remain extreme; always cross-check with trend analysis.
- rsi_6: RSI-6 (fast, 6-period): Highly responsive short-term momentum. Usage: Early entry/exit signals; overbought >70, oversold <30 (or tighter 80/20). Tips: Prone to whipsaws — cross-confirm with rsi_14 or rsi_21.
- rsi_14: RSI-14 (standard, 14-period): Industry-standard oscillator. Usage: Classic 70/30 thresholds and divergence analysis. Tips: Best balance of sensitivity vs noise. Equivalent to 'rsi'.
- rsi_21: RSI-21 (slow, 21-period): Smoothed long-term momentum. Usage: Identify sustained overbought/oversold regimes; divergence is a stronger signal. Tips: Best for swing/position trades. Use with rsi_6 for multi-timeframe confluence.

NOTE: For a richer momentum picture, consider requesting multiple RSI periods (e.g. rsi_6 + rsi_21) to observe short vs long-term momentum alignment — this is called RSI multi-timeframe confluence. When rsi_6 > rsi_21 and both are rising, momentum is strongly bullish. When they diverge, a reversal may be forming.

Volatility Indicators:
- boll: Bollinger Middle: A 20 SMA serving as the basis for Bollinger Bands. Usage: Acts as a dynamic benchmark for price movement. Tips: Combine with the upper and lower bands to effectively spot breakouts or reversals.
- boll_ub: Bollinger Upper Band: Typically 2 standard deviations above the middle line. Usage: Signals potential overbought conditions and breakout zones. Tips: Confirm signals with other tools; prices may ride the band in strong trends.
- boll_lb: Bollinger Lower Band: Typically 2 standard deviations below the middle line. Usage: Indicates potential oversold conditions. Tips: Use additional analysis to avoid false reversal signals.
- atr: ATR: Averages true range to measure volatility. Usage: Set stop-loss levels and adjust position sizes based on current market volatility. Tips: It's a reactive measure, so use it as part of a broader risk management strategy.

Volume-Based Indicators:
- vwma: VWMA: A moving average weighted by volume. Usage: Confirm trends by integrating price action with volume data. Tips: Watch for skewed results from volume spikes; use in combination with other volume analyses.

- Select indicators that provide diverse and complementary information. Avoid redundancy (e.g., do not select both rsi and stochrsi). Also briefly explain why they are suitable for the given market context. When you tool call, please use the exact name of the indicators provided above as they are defined parameters, otherwise your call will fail. Please make sure to call get_stock_data first to retrieve the CSV that is needed to generate indicators. Then use get_indicators with the specific indicator names. Write a very detailed and nuanced report of the trends you observe. Provide specific, actionable insights with supporting evidence to help traders make informed decisions."""
            + """ Make sure to append a Markdown table at the end of the report to organize key points in the report, organized and easy to read."""
            + get_language_instruction()
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful AI assistant, collaborating with other assistants."
                    " Use the provided tools to progress towards answering the question."
                    " If you are unable to fully answer, that's OK; another assistant with different tools"
                    " will help where you left off. Execute what you can to make progress."
                    " If you or any other assistant has the FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** or deliverable,"
                    " prefix your response with FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** so the team knows to stop."
                    " You have access to the following tools: {tool_names}.\n{system_message}"
                    "For your reference, the current date is {current_date}. {instrument_context}",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
        prompt = prompt.partial(current_date=current_date)
        prompt = prompt.partial(instrument_context=instrument_context)

        chain = prompt | llm.bind_tools(tools)

        result = chain.invoke(state["messages"])

        report = ""

        if len(result.tool_calls) == 0:
            report = result.content

        return {
            "messages": [result],
            "market_report": report,
        }

    return market_analyst_node
