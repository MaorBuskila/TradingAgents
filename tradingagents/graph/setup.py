# TradingAgents/graph/setup.py

from typing import Any, Dict, Optional
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from tradingagents.agents import *
from tradingagents.agents.utils.agent_states import AgentState
from tradingagents.dataflows.rsi_cache import get_rsi_params, upsert_rsi_params

from .conditional_logic import ConditionalLogic


def create_rsi_optimizer_node(config: Optional[Dict[str, Any]] = None):
    """
    Graph node that runs BEFORE any analyst.

    Logic:
      1. Check rsi_params_cache DB for this ticker.
      2. Cache HIT  → load params directly into AgentState (rsi_from_cache=True).
      3. Cache MISS → run the optimizer (algo or llm based on config),
                      store result in DB, write to AgentState.
      4. If optimizer result has LOW confidence → fall back to RSI-14 defaults
         but still persist them so re-runs skip the optimizer.
    """
    cfg = config or {}
    optimizer_provider = cfg.get("rsi_optimizer_provider", "algo")
    training_days      = cfg.get("rsi_training_days", 180)
    test_days          = cfg.get("rsi_test_days", 90)
    force_reoptimize   = cfg.get("rsi_force_reoptimize", False)

    def rsi_optimizer_node(state: AgentState) -> dict:
        ticker   = state["company_of_interest"].upper().strip()
        as_of    = state["trade_date"]

        # ── 1. Try cache first ──────────────────────────────────────────────
        if not force_reoptimize:
            cached = get_rsi_params(ticker)
            if cached:
                return {
                    "rsi_optimal_period": cached["optimal_period"],
                    "rsi_optimal_upper":  cached["optimal_upper"],
                    "rsi_optimal_lower":  cached["optimal_lower"],
                    "rsi_oos_sharpe":     cached["oos_sharpe"],
                    "rsi_is_sharpe":      cached["is_sharpe"],
                    "rsi_confidence":     cached["confidence"],
                    "rsi_regime":         cached.get("regime") or "unknown",
                    "rsi_from_cache":     True,
                }

        # ── 2. Cache miss → run optimizer ───────────────────────────────────
        result = None
        model_used = ""

        try:
            if optimizer_provider == "llm":
                from tradingagents.quant_ml.optimizers.rsi_optimizer_llm import optimize_rsi_llm
                llm_instance = cfg.get("llm_instance")
                result = optimize_rsi_llm(
                    ticker, as_of,
                    llm=llm_instance,
                    training_days=training_days,
                    test_days=test_days,
                )
                model_used = cfg.get("quick_think_llm", "")
            else:
                from tradingagents.quant_ml.optimizers.rsi_optimizer_algo import optimize_rsi
                result = optimize_rsi(
                    ticker, as_of,
                    training_days=training_days,
                    test_days=test_days,
                )
        except Exception as e:
            # Optimizer failed — use RSI-14 defaults and mark LOW confidence
            result = {
                "optimal_period": 14,
                "optimal_upper":  70.0,
                "optimal_lower":  30.0,
                "oos_sharpe":     0.0,
                "is_sharpe":      0.0,
                "confidence":     "LOW",
                "regime":         "unknown",
            }

        # ── 3. Downgrade LOW confidence to safe defaults ────────────────────
        if result.get("confidence") == "LOW":
            optimal_period = 14
            optimal_upper  = 70.0
            optimal_lower  = 30.0
        else:
            optimal_period = result["optimal_period"]
            optimal_upper  = result["optimal_upper"]
            optimal_lower  = result["optimal_lower"]

        # ── 4. Persist to DB ────────────────────────────────────────────────
        upsert_rsi_params(
            ticker             = ticker,
            optimal_period     = optimal_period,
            optimal_upper      = optimal_upper,
            optimal_lower      = optimal_lower,
            oos_sharpe         = result.get("oos_sharpe", 0.0),
            is_sharpe          = result.get("is_sharpe", 0.0),
            confidence         = result.get("confidence", "LOW"),
            regime             = result.get("regime", "unknown"),
            training_days      = training_days,
            test_days          = test_days,
            optimizer_provider = optimizer_provider,
            model_used         = model_used,
        )

        return {
            "rsi_optimal_period": optimal_period,
            "rsi_optimal_upper":  optimal_upper,
            "rsi_optimal_lower":  optimal_lower,
            "rsi_oos_sharpe":     result.get("oos_sharpe", 0.0),
            "rsi_is_sharpe":      result.get("is_sharpe", 0.0),
            "rsi_confidence":     result.get("confidence", "LOW"),
            "rsi_regime":         result.get("regime", "unknown"),
            "rsi_from_cache":     False,
        }

    return rsi_optimizer_node


class GraphSetup:
    """Handles the setup and configuration of the agent graph."""

    def __init__(
        self,
        quick_thinking_llm: Any,
        deep_thinking_llm: Any,
        tool_nodes: Dict[str, ToolNode],
        conditional_logic: ConditionalLogic,
    ):
        """Initialize with required components."""
        self.quick_thinking_llm = quick_thinking_llm
        self.deep_thinking_llm = deep_thinking_llm
        self.tool_nodes = tool_nodes
        self.conditional_logic = conditional_logic

    def setup_graph(
        self,
        selected_analysts=["market", "social", "news", "fundamentals"],
        config: Optional[Dict[str, Any]] = None,
    ):
        """Set up and compile the agent workflow graph.

        Args:
            selected_analysts (list): List of analyst types to include. Options are:
                - "market": Market analyst
                - "social": Social media analyst
                - "news": News analyst
                - "fundamentals": Fundamentals analyst
        """
        if len(selected_analysts) == 0:
            raise ValueError("Trading Agents Graph Setup Error: no analysts selected!")

        # RSI Optimizer node — runs once before all analysts
        rsi_optimizer_node = create_rsi_optimizer_node(config or {})

        # Create analyst nodes
        analyst_nodes = {}
        delete_nodes = {}
        tool_nodes = {}

        if "market" in selected_analysts:
            analyst_nodes["market"] = create_market_analyst(
                self.quick_thinking_llm
            )
            delete_nodes["market"] = create_msg_delete()
            tool_nodes["market"] = self.tool_nodes["market"]

        if "social" in selected_analysts:
            analyst_nodes["social"] = create_social_media_analyst(
                self.quick_thinking_llm
            )
            delete_nodes["social"] = create_msg_delete()
            tool_nodes["social"] = self.tool_nodes["social"]

        if "news" in selected_analysts:
            analyst_nodes["news"] = create_news_analyst(
                self.quick_thinking_llm
            )
            delete_nodes["news"] = create_msg_delete()
            tool_nodes["news"] = self.tool_nodes["news"]

        if "fundamentals" in selected_analysts:
            analyst_nodes["fundamentals"] = create_fundamentals_analyst(
                self.quick_thinking_llm
            )
            delete_nodes["fundamentals"] = create_msg_delete()
            tool_nodes["fundamentals"] = self.tool_nodes["fundamentals"]

        # Create researcher and manager nodes
        bull_researcher_node = create_bull_researcher(self.quick_thinking_llm)
        bear_researcher_node = create_bear_researcher(self.quick_thinking_llm)
        research_manager_node = create_research_manager(self.deep_thinking_llm)
        trader_node = create_trader(self.quick_thinking_llm)

        # Create risk analysis nodes
        aggressive_analyst = create_aggressive_debator(self.quick_thinking_llm)
        neutral_analyst = create_neutral_debator(self.quick_thinking_llm)
        conservative_analyst = create_conservative_debator(self.quick_thinking_llm)
        portfolio_manager_node = create_portfolio_manager(self.deep_thinking_llm)

        # Create workflow
        workflow = StateGraph(AgentState)

        # Add RSI Optimizer as the very first node
        workflow.add_node("RSI Optimizer", rsi_optimizer_node)

        # Add analyst nodes to the graph
        for analyst_type, node in analyst_nodes.items():
            workflow.add_node(f"{analyst_type.capitalize()} Analyst", node)
            workflow.add_node(
                f"Msg Clear {analyst_type.capitalize()}", delete_nodes[analyst_type]
            )
            workflow.add_node(f"tools_{analyst_type}", tool_nodes[analyst_type])

        # Add other nodes
        workflow.add_node("Bull Researcher", bull_researcher_node)
        workflow.add_node("Bear Researcher", bear_researcher_node)
        workflow.add_node("Research Manager", research_manager_node)
        workflow.add_node("Trader", trader_node)
        workflow.add_node("Aggressive Analyst", aggressive_analyst)
        workflow.add_node("Neutral Analyst", neutral_analyst)
        workflow.add_node("Conservative Analyst", conservative_analyst)
        workflow.add_node("Portfolio Manager", portfolio_manager_node)

        # Define edges
        # START → RSI Optimizer → first analyst
        first_analyst = selected_analysts[0]
        workflow.add_edge(START, "RSI Optimizer")
        workflow.add_edge("RSI Optimizer", f"{first_analyst.capitalize()} Analyst")

        # Connect analysts in sequence
        for i, analyst_type in enumerate(selected_analysts):
            current_analyst = f"{analyst_type.capitalize()} Analyst"
            current_tools = f"tools_{analyst_type}"
            current_clear = f"Msg Clear {analyst_type.capitalize()}"

            # Add conditional edges for current analyst
            workflow.add_conditional_edges(
                current_analyst,
                getattr(self.conditional_logic, f"should_continue_{analyst_type}"),
                [current_tools, current_clear],
            )
            workflow.add_edge(current_tools, current_analyst)

            # Connect to next analyst or to Bull Researcher if this is the last analyst
            if i < len(selected_analysts) - 1:
                next_analyst = f"{selected_analysts[i+1].capitalize()} Analyst"
                workflow.add_edge(current_clear, next_analyst)
            else:
                workflow.add_edge(current_clear, "Bull Researcher")

        # Add remaining edges
        workflow.add_conditional_edges(
            "Bull Researcher",
            self.conditional_logic.should_continue_debate,
            {
                "Bear Researcher": "Bear Researcher",
                "Research Manager": "Research Manager",
            },
        )
        workflow.add_conditional_edges(
            "Bear Researcher",
            self.conditional_logic.should_continue_debate,
            {
                "Bull Researcher": "Bull Researcher",
                "Research Manager": "Research Manager",
            },
        )
        workflow.add_edge("Research Manager", "Trader")
        workflow.add_edge("Trader", "Aggressive Analyst")
        workflow.add_conditional_edges(
            "Aggressive Analyst",
            self.conditional_logic.should_continue_risk_analysis,
            {
                "Conservative Analyst": "Conservative Analyst",
                "Portfolio Manager": "Portfolio Manager",
            },
        )
        workflow.add_conditional_edges(
            "Conservative Analyst",
            self.conditional_logic.should_continue_risk_analysis,
            {
                "Neutral Analyst": "Neutral Analyst",
                "Portfolio Manager": "Portfolio Manager",
            },
        )
        workflow.add_conditional_edges(
            "Neutral Analyst",
            self.conditional_logic.should_continue_risk_analysis,
            {
                "Aggressive Analyst": "Aggressive Analyst",
                "Portfolio Manager": "Portfolio Manager",
            },
        )

        workflow.add_edge("Portfolio Manager", END)

        return workflow
