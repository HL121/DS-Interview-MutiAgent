"""End-to-end check of the Streamlit demo with Streamlit's AppTest, a fake pipeline and a
scripted LLM (no browser, no API calls).

Run: python tests/test_demo.py   (or: pytest tests/test_demo.py)
"""

import os
import re
import sys
import tempfile
from pathlib import Path

_TMP = tempfile.mkdtemp()
os.environ["LONG_TERM_DB"] = os.path.join(_TMP, "long_term.db")
os.environ["CHECKPOINT_DB"] = os.path.join(_TMP, "checkpoints.db")
os.environ.setdefault("OPENAI_API_KEY", "dummy")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from langchain_core.messages import AIMessage  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402

import orchestration  # noqa: E402
from scripts.controller import graph as g  # noqa: E402

SQL, CLS = "sql_window_function", "classification"


class FakeSkill:
    def __init__(self, client): pass
    def run(self, jd_text, user_desc): return {"extracted": {}, "mapped": {}, "weights": {SQL: 0.6, CLS: 0.4}}


class FakeScope:
    def __init__(self, client): pass
    def run(self, skill_weights, days_left, **kw):
        return {"total_questions": 8, "difficulty_distribution": {"easy": 0.5, "medium": 0.5, "hard": 0.0},
                "skill_plan": {SQL: 4, CLS: 4}}


class FakeRetriever:
    def retrieve(self, query, topk, **kw):
        return [{"id": f"{query}_{i}", "title": f"{query} q{i}", "type": "theory", "category": "ML",
                 "difficulty": "easy" if i % 2 else "medium", "taxonomy_skills": [query],
                 "selection_reason": f"matches {query}"} for i in range(topk)]


def fake_plan(tasks, days_left, skill_plan, **kw):
    chosen = []
    for skill, n in skill_plan.items():
        chosen += [dict(t) for t in tasks if t["requested_skill"] == skill][:n]
    return [chosen[i::days_left] for i in range(days_left)], [{"day": i + 1, "summary": f"summary {i + 1}"} for i in range(days_left)]


orchestration.SkillAnalyzerAgent, orchestration.ScopePlannerAgent = FakeSkill, FakeScope
orchestration.init_agentic_retriever = lambda llm=None: FakeRetriever()
orchestration.normalize_tasks = lambda t: t
orchestration.run_planning_agent = fake_plan


class ScriptedLLM:
    def __init__(self):
        self.script = []

    def bind_tools(self, tools, **kwargs):
        return self

    def invoke(self, messages):
        return self.script.pop(0)


LLM = ScriptedLLM()
g.LLM = LLM


def labels(elements):
    return [e.label for e in elements]


def day_labels(at):
    """Labels of the day panels in day order. AppTest lists an expander that has an icon
    (today's panel) under at.status, so both lists are merged."""
    panels = [e.label for e in at.expander] + [s.label for s in at.status]
    days = [label for label in panels if label.startswith("第 ")]
    return sorted(days, key=lambda label: int(re.match(r"第 (\d+) 天", label).group(1)))


def test_demo_flow():
    at = AppTest.from_file(str(ROOT / "demo.py"), default_timeout=60).run()
    assert not at.exception, at.exception
    assert at.title[0].value == "新建学习计划"

    # 1. Generate a plan: the form runs the pipeline and switches to the plan page.
    at.text_area[0].input("Data Scientist, Product Analytics\nSQL and ML")
    at.text_area[1].input("Statistics student")
    at.slider[0].set_value(4)
    next(b for b in at.button if b.label == "生成计划").click()
    at.run()
    assert not at.exception, at.exception
    assert at.title[0].value == "Data Scientist, Product Analytics"
    assert len(day_labels(at)) == 4 and "今天" in day_labels(at)[0], day_labels(at)
    thread_id = at.session_state["thread_id"]
    assert any("Data Scientist" in label for label in labels(at.sidebar.button)), "plan listed in the sidebar"

    # AppTest keeps the form's widgets from before st.rerun(), which breaks the next run. Continue in a
    # fresh session opened on this plan, as a user would by clicking it in the sidebar; this also checks
    # that a saved plan reloads from the checkpoint DB.
    at = AppTest.from_file(str(ROOT / "demo.py"), default_timeout=60)
    at.session_state["thread_id"] = thread_id
    at.run()
    assert not at.exception, at.exception
    assert len(day_labels(at)) == 4

    # 2. Mark today's first question wrong: status saved, memory panel shows the weak skill.
    first = g.get_state(thread_id)["days"][0][0]
    at.button(key=f"wrong-{first['id']}").click()
    at.run()
    assert not at.exception, at.exception
    assert g.get_state(thread_id)["days"][0][0]["status"] == "wrong"
    assert any(first["requested_skill"] in m.value for m in at.sidebar.markdown), "weak skill shown in memory panel"

    # 3. Chat: the controller calls update_plan; the reply, the tool step and the new questions show up.
    LLM.script = [
        AIMessage(content="", tool_calls=[{"name": "update_plan", "id": "c1",
                                           "args": {"reason": "user is weak in SQL", "struggling_skills": [SQL]}}]),
        AIMessage(content="从明天开始加了 2 道窗口函数练习。"),
    ]
    at.chat_input[0].set_value("我实在不会 SQL 窗口函数")
    at.run()
    assert not at.exception, at.exception
    assert any("从明天开始加了 2 道窗口函数练习" in m.value for m in at.markdown)
    assert any("update_plan" in c.value for c in at.caption), "tool call shown"
    assert any("新增" in label for label in day_labels(at)), "days with new questions are marked"

    # 4. Simulate the next day: today moves to day 2.
    next(b for b in at.sidebar.button if b.label == "模拟：进入下一天").click()
    at.run()
    assert not at.exception, at.exception
    assert "今天" in day_labels(at)[1] and "今天" not in day_labels(at)[0], day_labels(at)


if __name__ == "__main__":
    test_demo_flow()
    print("PASS test_demo_flow\n\n1 test passed")
