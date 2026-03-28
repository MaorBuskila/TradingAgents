from typing import Dict, Any, List, Optional, Callable
import time
from pathlib import Path
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
import ast

from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.analysis_selections import ANALYST_ORDER
ANALYST_AGENT_NAMES = {
    "market": "Market Analyst",
    "social": "Social Analyst",
    "news": "News Analyst",
    "fundamentals": "Fundamentals Analyst",
}
ANALYST_REPORT_MAP = {
    "market": "market_report",
    "social": "sentiment_report",
    "news": "news_report",
    "fundamentals": "fundamentals_report",
}

def extract_content_string(content):
    """Extract string content from various message formats."""
    def is_empty(val):
        if val is None or val == '':
            return True
        if isinstance(val, str):
            s = val.strip()
            if not s:
                return True
            try:
                return not bool(ast.literal_eval(s))
            except (ValueError, SyntaxError):
                return False
        return not bool(val)

    if is_empty(content):
        return None

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, dict):
        text = content.get('text', '')
        return text.strip() if not is_empty(text) else None

    if isinstance(content, list):
        text_parts = [
            item.get('text', '').strip() if isinstance(item, dict) and item.get('type') == 'text'
            else (item.strip() if isinstance(item, str) else '')
            for item in content
        ]
        result = ' '.join(t for t in text_parts if t and not is_empty(t))
        return result if result else None

    return str(content).strip() if not is_empty(content) else None


def classify_message_type(message) -> tuple[str, str | None]:
    """Classify LangChain message into display type and extract content."""
    content = extract_content_string(getattr(message, 'content', None))

    if isinstance(message, HumanMessage):
        if content and content.strip() == "Continue":
            return ("Control", content)
        return ("User", content)

    if isinstance(message, ToolMessage):
        return ("Data", content)

    if isinstance(message, AIMessage):
        return ("Agent", content)

    return ("System", content)


class AnalysisRunner:
    def __init__(
        self,
        selections: Dict[str, Any],
        callbacks: Optional[List[Any]] = None,
        on_message: Optional[Callable] = None,
        on_tool_call: Optional[Callable] = None,
        on_agent_status: Optional[Callable] = None,
        on_report_section: Optional[Callable] = None,
        on_complete: Optional[Callable] = None,
    ):
        self.selections = selections
        self.callbacks = callbacks or []
        self.on_message = on_message
        self.on_tool_call = on_tool_call
        self.on_agent_status = on_agent_status
        self.on_report_section = on_report_section
        self.on_complete = on_complete

        # State tracking
        self.report_sections = {}
        self.agent_status = {}
        self._last_message_id = None
        self.selected_analysts = [a.value if hasattr(a, 'value') else a for a in selections.get("analysts", [])]
        self.selected_analyst_keys = [a for a in ANALYST_ORDER if a in self.selected_analysts]

        # Initialize agent statuses
        for analyst_key in self.selected_analyst_keys:
            if analyst_key in ANALYST_AGENT_NAMES:
                self.agent_status[ANALYST_AGENT_NAMES[analyst_key]] = "pending"
        
        fixed_teams = {
            "Research Team": ["Bull Researcher", "Bear Researcher", "Research Manager"],
            "Trading Team": ["Trader"],
            "Risk Management": ["Aggressive Analyst", "Neutral Analyst", "Conservative Analyst"],
            "Portfolio Management": ["Portfolio Manager"],
        }
        for team_agents in fixed_teams.values():
            for agent in team_agents:
                self.agent_status[agent] = "pending"

    def _update_agent_status(self, agent: str, status: str):
        if self.agent_status.get(agent) != status:
            self.agent_status[agent] = status
            if self.on_agent_status:
                self.on_agent_status(agent, status)

    def _update_report_section(self, section: str, content: str):
        if self.report_sections.get(section) != content:
            self.report_sections[section] = content
            if self.on_report_section:
                self.on_report_section(section, content)

    def _update_research_team_status(self, status: str):
        research_team = ["Bull Researcher", "Bear Researcher", "Research Manager"]
        for agent in research_team:
            self._update_agent_status(agent, status)

    def _update_analyst_statuses(self, chunk: Dict[str, Any]):
        found_active = False

        for analyst_key in ANALYST_ORDER:
            if analyst_key not in self.selected_analyst_keys:
                continue

            agent_name = ANALYST_AGENT_NAMES[analyst_key]
            report_key = ANALYST_REPORT_MAP[analyst_key]

            if chunk.get(report_key):
                self._update_report_section(report_key, chunk[report_key])

            has_report = bool(self.report_sections.get(report_key))

            if has_report:
                self._update_agent_status(agent_name, "completed")
            elif not found_active:
                self._update_agent_status(agent_name, "in_progress")
                found_active = True
            else:
                self._update_agent_status(agent_name, "pending")

        if not found_active and self.selected_analyst_keys:
            if self.agent_status.get("Bull Researcher") == "pending":
                self._update_agent_status("Bull Researcher", "in_progress")

    def run(self):
        config = DEFAULT_CONFIG.copy()
        config["max_debate_rounds"] = self.selections["research_depth"]
        config["max_risk_discuss_rounds"] = self.selections["research_depth"]
        config["quick_think_llm"] = self.selections["shallow_thinker"]
        config["deep_think_llm"] = self.selections["deep_thinker"]
        config["backend_url"] = self.selections["backend_url"]
        config["llm_provider"] = self.selections["llm_provider"].lower()
        config["google_thinking_level"] = self.selections.get("google_thinking_level")
        config["openai_reasoning_effort"] = self.selections.get("openai_reasoning_effort")
        config["anthropic_effort"] = self.selections.get("anthropic_effort")

        graph = TradingAgentsGraph(
            self.selected_analyst_keys,
            config=config,
            debug=True,
            callbacks=self.callbacks,
        )

        init_agent_state = graph.propagator.create_initial_state(
            self.selections["ticker"], self.selections["analysis_date"]
        )
        args = graph.propagator.get_graph_args(callbacks=self.callbacks)

        trace = []
        for chunk in graph.graph.stream(init_agent_state, **args):
            if len(chunk["messages"]) > 0:
                last_message = chunk["messages"][-1]
                msg_id = getattr(last_message, "id", None)

                if msg_id != self._last_message_id:
                    self._last_message_id = msg_id

                    msg_type, content = classify_message_type(last_message)
                    if content and content.strip() and self.on_message:
                        self.on_message(msg_type, content)

                    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                        for tool_call in last_message.tool_calls:
                            if self.on_tool_call:
                                if isinstance(tool_call, dict):
                                    self.on_tool_call(tool_call["name"], tool_call["args"])
                                else:
                                    self.on_tool_call(tool_call.name, tool_call.args)

            self._update_analyst_statuses(chunk)

            # Research Team
            if chunk.get("investment_debate_state"):
                debate_state = chunk["investment_debate_state"]
                bull_hist = debate_state.get("bull_history", "").strip()
                bear_hist = debate_state.get("bear_history", "").strip()
                judge = debate_state.get("judge_decision", "").strip()

                if bull_hist or bear_hist:
                    self._update_research_team_status("in_progress")
                if bull_hist:
                    self._update_report_section("investment_plan", f"### Bull Researcher Analysis\n{bull_hist}")
                if bear_hist:
                    self._update_report_section("investment_plan", f"### Bear Researcher Analysis\n{bear_hist}")
                if judge:
                    self._update_report_section("investment_plan", f"### Research Manager Decision\n{judge}")
                    self._update_research_team_status("completed")
                    self._update_agent_status("Trader", "in_progress")

            # Trading Team
            if chunk.get("trader_investment_plan"):
                self._update_report_section("trader_investment_plan", chunk["trader_investment_plan"])
                if self.agent_status.get("Trader") != "completed":
                    self._update_agent_status("Trader", "completed")
                    self._update_agent_status("Aggressive Analyst", "in_progress")

            # Risk Management Team
            if chunk.get("risk_debate_state"):
                risk_state = chunk["risk_debate_state"]
                agg_hist = risk_state.get("aggressive_history", "").strip()
                con_hist = risk_state.get("conservative_history", "").strip()
                neu_hist = risk_state.get("neutral_history", "").strip()
                judge = risk_state.get("judge_decision", "").strip()

                if agg_hist:
                    if self.agent_status.get("Aggressive Analyst") != "completed":
                        self._update_agent_status("Aggressive Analyst", "in_progress")
                    self._update_report_section("final_trade_decision", f"### Aggressive Analyst Analysis\n{agg_hist}")
                if con_hist:
                    if self.agent_status.get("Conservative Analyst") != "completed":
                        self._update_agent_status("Conservative Analyst", "in_progress")
                    self._update_report_section("final_trade_decision", f"### Conservative Analyst Analysis\n{con_hist}")
                if neu_hist:
                    if self.agent_status.get("Neutral Analyst") != "completed":
                        self._update_agent_status("Neutral Analyst", "in_progress")
                    self._update_report_section("final_trade_decision", f"### Neutral Analyst Analysis\n{neu_hist}")
                if judge:
                    if self.agent_status.get("Portfolio Manager") != "completed":
                        self._update_agent_status("Portfolio Manager", "in_progress")
                        self._update_report_section("final_trade_decision", f"### Portfolio Manager Decision\n{judge}")
                        self._update_agent_status("Aggressive Analyst", "completed")
                        self._update_agent_status("Conservative Analyst", "completed")
                        self._update_agent_status("Neutral Analyst", "completed")
                        self._update_agent_status("Portfolio Manager", "completed")

            trace.append(chunk)

        final_state = trace[-1] if trace else {}
        decision = graph.process_signal(final_state.get("final_trade_decision", "")) if "final_trade_decision" in final_state else None

        for agent in self.agent_status:
            self._update_agent_status(agent, "completed")

        if self.on_message:
            self.on_message("System", f"Completed analysis for {self.selections['analysis_date']}")

        for section in self.report_sections.keys():
            if section in final_state:
                self._update_report_section(section, final_state[section])

        if self.on_complete:
            self.on_complete(final_state, decision)

        return final_state, decision
