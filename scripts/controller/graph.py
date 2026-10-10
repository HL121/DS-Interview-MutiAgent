"""The agent graph: first plan via the fixed pipeline, follow-up messages via a ReAct controller.

START -> plan exists? -- no  --> agent1 -> agent_scope -> agent2 -> agent3 -> END
                      -- yes --> controller <-> tools (until no tool call) -> END

Guardrails live in code: at most MAX_TOOL_CALLS tool calls per user turn, the expensive
analyze_skills / generate_plan at most once per turn, and no parallel tool calls (each
tool must see the state left by the previous one).
"""

import uuid
from datetime import date
from pathlib import Path
from typing import Annotated, Any, Dict, Optional

from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage, ToolMessage, trim_messages
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from orchestration import (
    DEFAULT_USER, AgentState, _plan_output, agent1, agent2, agent3, agent_scope, checkpointer, client,
)
from scripts.controller.feedback import skill_factor
from scripts.controller.memory import get_preferences
from scripts.controller.replan import today_index
from scripts.controller.tools import TAXONOMY, TOOLS

LLM = client
MAX_TOOL_CALLS = 5
ONCE_PER_TURN = {"analyze_skills", "generate_plan"}
MAX_HISTORY = 10  # messages kept from earlier turns; the current turn is always kept whole
WEEKDAYS_ZH = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
SYSTEM_PROMPT = (
    (Path(__file__).resolve().parents[2] / "prompts" / "controller_system.txt")
    .read_text()
    .replace("{taxonomy}", ", ".join(sorted(TAXONOMY)))
)


class AppState(AgentState):
    messages: Annotated[list[AnyMessage], add_messages]


def _turn_start(messages: list) -> int:
    """Index of the latest user message; everything after it belongs to the current turn."""
    return max((i for i, m in enumerate(messages) if isinstance(m, HumanMessage)), default=0)


def available_tools(messages: list) -> list:
    used = [m.name for m in messages[_turn_start(messages):] if isinstance(m, ToolMessage)]
    if len(used) >= MAX_TOOL_CALLS:
        return []  # budget spent: the controller must answer without tools
    return [t for t in TOOLS if not (t.name in ONCE_PER_TURN and t.name in used)]


def render_context(state: dict) -> str:
    """Compact view of the state for the LLM, instead of the full plan JSON."""
    days = state.get("days") or []
    user = state.get("user_id") or DEFAULT_USER
    today = min(today_index(state), len(days) - 1)
    weights = state.get("weights") or {}

    progress = {}
    for day in days:
        for t in day:
            done, total = progress.get(t.get("requested_skill"), (0, 0))
            progress[t.get("requested_skill")] = (done + (t.get("status") != "todo"), total + 1)

    lines = [
        f"Today: {date.today().isoformat()} ({date.today():%A} {WEEKDAYS_ZH[date.today().weekday()]}), plan day {today + 1} of {len(days)}.",
        "Skill weights: " + ", ".join(f"{s} {w:.2f}" for s, w in sorted(weights.items(), key=lambda x: -x[1])),
        "Progress (finished/total): " + ", ".join(f"{s} {d}/{n}" for s, (d, n) in progress.items()),
        f"Weak skills (from history): {[s for s in weights if skill_factor(user, s) > 1] or 'none'}",
        f"Mastered skills: {[s for s in weights if skill_factor(user, s) < 1] or 'none'}",
        f"Preferences: {get_preferences(user) or 'none'}",
    ]
    earlier = [t for day in days[:today] for t in day if t.get("status") == "todo"]
    if earlier:
        lines.append("Unfinished from earlier days: " + "; ".join(f"{t['id']} ({t['title'][:50]})" for t in earlier))
    for i in range(today, min(today + 3, len(days))):
        lines.append(f"Day {i + 1}{' (today)' if i == today else ''}  [no. | id | title | skill | difficulty | status | why chosen]")
        # Numbered, so "the second question" is a lookup rather than counting.
        for no, t in enumerate(days[i], start=1):
            lines.append(
                f"{no}. {t['id']} | {t['title'][:60]} | {t.get('requested_skill')} | {t.get('difficulty')} | "
                f"{t.get('status', 'todo')} | {(t.get('selection_reason') or '')[:100]}"
            )
    for i in range(today + 3, len(days)):
        diffs = [t.get("difficulty") for t in days[i]]
        lines.append(f"Day {i + 1}: {len(days[i])} questions, " + ", ".join(f"{diffs.count(d)} {d}" for d in ("easy", "medium", "hard") if d in diffs))
    return "\n".join(lines)


def controller(state: dict) -> dict:
    messages = state["messages"]
    start = _turn_start(messages)
    # Earlier turns are trimmed to the last MAX_HISTORY messages, starting at a user message so
    # a tool call is never separated from its result. The current turn is kept whole.
    earlier = trim_messages(messages[:start], strategy="last", token_counter=len, max_tokens=MAX_HISTORY, start_on="human")
    tools = available_tools(messages)
    llm = LLM.bind_tools(tools, parallel_tool_calls=False) if tools else LLM
    # Fixed instructions first, so the provider can cache that prefix; the changing context after it.
    prompt = [SystemMessage(SYSTEM_PROMPT), SystemMessage(render_context(state)), *earlier, *messages[start:]]
    return {"messages": [llm.invoke(prompt)]}


graph = StateGraph(AppState)
for name, node in [("agent1", agent1), ("agent_scope", agent_scope), ("agent2", agent2), ("agent3", agent3)]:
    graph.add_node(name, node)
graph.add_node("controller", controller)
# handle_tool_errors=True: a failing tool (e.g. Qdrant unreachable) returns the error to the LLM
# instead of crashing the turn.
graph.add_node("tools", ToolNode(TOOLS, handle_tool_errors=True))

graph.add_conditional_edges(START, lambda s: "controller" if s.get("days") else "agent1", ["controller", "agent1"])
graph.add_edge("agent1", "agent_scope")
graph.add_edge("agent_scope", "agent2")
graph.add_edge("agent2", "agent3")
graph.add_edge("agent3", END)
graph.add_conditional_edges("controller", tools_condition)  # tool calls -> "tools", otherwise END
graph.add_edge("tools", "controller")

app = graph.compile(checkpointer=checkpointer)


def _config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


def start_plan(jd: str, user_desc: str, days_left: int, thread_id: Optional[str] = None, user_id: str = DEFAULT_USER) -> Dict[str, Any]:
    """First turn: build the initial plan with the fixed pipeline."""
    thread_id = thread_id or str(uuid.uuid4())
    app.invoke(
        {"jd": jd, "user_desc": user_desc, "days_left": days_left, "user_id": user_id,
         "thread_id": thread_id, "start_date": date.today().isoformat()},
        _config(thread_id),
    )
    return load_plan(thread_id)


def chat(thread_id: str, message: str) -> str:
    """Follow-up turn: the controller decides which tools to call. Returns its reply."""
    result = app.invoke({"messages": [HumanMessage(message)]}, _config(thread_id))
    return result["messages"][-1].content


def load_plan(thread_id: str) -> Optional[Dict[str, Any]]:
    state = app.get_state(_config(thread_id)).values
    return _plan_output(state, thread_id) if state else None


if __name__ == "__main__":
    # Terminal chat for testing: python -m scripts.controller.graph <thread_id>
    import sys

    thread = sys.argv[1]
    print(f"Chatting on thread {thread}. Empty line to quit.")
    while message := input("you> ").strip():
        seen = len(app.get_state(_config(thread)).values.get("messages") or [])
        reply = chat(thread, message)
        for m in app.get_state(_config(thread)).values["messages"][seen + 1:-1]:
            for call in getattr(m, "tool_calls", None) or []:
                print(f"  [tool call]   {call['name']}({call['args']})")
            if m.type == "tool":
                print(f"  [tool result] {m.content}")
        print("bot>", reply)
