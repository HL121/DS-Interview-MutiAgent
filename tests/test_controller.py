"""Graph wiring and guardrail checks for the controller, with a scripted fake LLM (no API calls).

Run: python tests/test_controller.py   (or: pytest tests/test_controller.py)
"""

import os
import sys
import tempfile
from pathlib import Path

_TMP = tempfile.mkdtemp()
os.environ["LONG_TERM_DB"] = os.path.join(_TMP, "long_term.db")
os.environ["CHECKPOINT_DB"] = os.path.join(_TMP, "checkpoints.db")
os.environ.setdefault("OPENAI_API_KEY", "dummy")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage  # noqa: E402

import orchestration  # noqa: E402
from scripts.controller import graph as g  # noqa: E402

SQL, CLS = "sql_window_function", "classification"


# ---- fake pipeline modules (deterministic) ----
class FakeSkill:
    def __init__(self, client): pass
    def run(self, jd_text, user_desc): return {"extracted": {}, "mapped": {}, "weights": {SQL: 0.6, CLS: 0.4}}


class FakeScope:
    def __init__(self, client): pass
    def run(self, skill_weights, days_left, **kw):
        return {"total_questions": 8, "difficulty_distribution": {"easy": 0.5, "medium": 0.5, "hard": 0.0},
                "skill_plan": {SQL: 4, CLS: 4}}


class FakeRetriever:
    fail = False
    def retrieve(self, query, topk, **kw):
        if FakeRetriever.fail:
            raise ConnectionError("Qdrant unreachable")
        return [{"id": f"{query}_{i}", "title": f"{query} q{i}", "type": "theory", "difficulty": "easy" if i % 2 else "medium",
                 "taxonomy_skills": [query], "selection_reason": f"matches {query}"} for i in range(topk)]


def fake_plan(tasks, days_left, skill_plan, **kw):
    chosen = []
    for skill, n in skill_plan.items():
        chosen += [dict(t) for t in tasks if t["requested_skill"] == skill][:n]
    return [chosen[i::days_left] for i in range(days_left)], [{"day": i + 1, "summary": ""} for i in range(days_left)]


orchestration.SkillAnalyzerAgent, orchestration.ScopePlannerAgent = FakeSkill, FakeScope
orchestration.init_agentic_retriever = lambda llm=None: FakeRetriever()
orchestration.normalize_tasks = lambda t: t
orchestration.run_planning_agent = fake_plan


class ScriptedLLM:
    """Returns pre-written AI messages and records what each call could see."""
    def __init__(self):
        self.script, self.calls, self._bound = [], [], None

    def bind_tools(self, tools, **kwargs):
        self._bound = ([t.name for t in tools], kwargs)
        return self

    def invoke(self, messages):
        tools, kwargs = self._bound or ([], {})
        self.calls.append({"tools": tools, "kwargs": kwargs, "messages": messages})
        self._bound = None
        return self.script.pop(0)


def tool_call(name, args, cid):
    return AIMessage(content="", tool_calls=[{"name": name, "args": {"reason": "test", **args}, "id": cid}])


def new_thread(name):
    llm = ScriptedLLM()
    g.LLM = llm
    g.start_plan("jd", "me", 4, thread_id=name, user_id=f"user_{name}")
    return llm


def state_of(thread):
    return g.app.get_state(g._config(thread)).values


def test_first_turn_runs_pipeline_without_llm():
    llm = new_thread("first")
    plan = g.load_plan("first")
    assert len(plan["days"]) == 4 and all(t["status"] == "todo" for d in plan["days"] for t in d)
    assert llm.calls == [] and not state_of("first").get("messages")


def test_feedback_turn_calls_update_plan():
    llm = new_thread("feedback")
    before = state_of("feedback")
    llm.script = [tool_call("update_plan", {"struggling_skills": [SQL]}, "c1"), AIMessage(content="Added 2 SQL questions.")]
    reply = g.chat("feedback", "I really don't get SQL window functions")
    after = state_of("feedback")

    assert reply == "Added 2 SQL questions."
    assert after["weights"][SQL] > before["weights"][SQL], "weight should rise"
    before_ids = {t["id"] for d in before["days"] for t in d}
    assert len({t["id"] for d in after["days"] for t in d} - before_ids) == 2, "2 reinforcement questions expected"
    first = llm.calls[0]
    assert first["kwargs"] == {"parallel_tool_calls": False}
    assert sorted(first["tools"]) == sorted(t.name for t in g.TOOLS)
    context = first["messages"][1].content
    assert isinstance(first["messages"][1], SystemMessage) and "(today)" in context and "why chosen" in context
    assert SQL in first["messages"][0].content, "taxonomy should be in the system prompt"
    assert any(isinstance(m, ToolMessage) and m.name == "update_plan" for m in after["messages"])


def test_tool_budget():
    llm = new_thread("budget")
    llm.script = [tool_call("retrieve_questions", {"skills": {SQL: 1}}, f"c{i}") for i in range(5)]
    llm.script.append(AIMessage(content="done"))
    g.chat("budget", "find more SQL")
    tool_msgs = [m for m in state_of("budget")["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_msgs) == g.MAX_TOOL_CALLS
    assert llm.calls[-1]["tools"] == [], "after the budget is spent no tools are offered"
    assert all(c["tools"] for c in llm.calls[:-1])


def test_expensive_tools_once_per_turn():
    llm = new_thread("once")
    llm.script = [tool_call("generate_plan", {}, "c1"), AIMessage(content="rebuilt")]
    g.chat("once", "I switched to a new role")
    assert "generate_plan" not in llm.calls[1]["tools"]
    assert "update_plan" in llm.calls[1]["tools"]
    llm.script = [AIMessage(content="ok")]
    g.chat("once", "thanks")
    assert "generate_plan" in llm.calls[2]["tools"], "a new turn resets the limit"


def test_history_trimmed_but_current_turn_kept():
    llm = new_thread("history")
    for i in range(8):
        llm.script = [AIMessage(content=f"answer {i}")]
        g.chat("history", f"question {i}")
    llm.script = [tool_call("update_plan", {}, "c1"), AIMessage(content="final")]
    g.chat("history", "last question")
    for call in llm.calls[-2:]:
        convo = [m for m in call["messages"] if not isinstance(m, SystemMessage)]
        start = max(i for i, m in enumerate(convo) if isinstance(m, HumanMessage) and m.content == "last question")
        assert len(convo[:start]) <= g.MAX_HISTORY
        assert isinstance(convo[0], HumanMessage), "history must start at a user message"
    last = [m for m in llm.calls[-1]["messages"] if not isinstance(m, SystemMessage)]
    assert isinstance(last[-1], ToolMessage), "current turn (incl. tool result) is kept whole"


def test_tool_error_goes_back_to_llm():
    llm = new_thread("error")
    FakeRetriever.fail = True
    try:
        llm.script = [tool_call("retrieve_questions", {"skills": {SQL: 2}}, "c1"), AIMessage(content="Search is down, try later.")]
        reply = g.chat("error", "more SQL please")
    finally:
        FakeRetriever.fail = False
    err = [m for m in state_of("error")["messages"] if isinstance(m, ToolMessage)][-1]
    assert "Qdrant unreachable" in err.content and reply == "Search is down, try later."


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"\n{len(tests)} tests passed")
