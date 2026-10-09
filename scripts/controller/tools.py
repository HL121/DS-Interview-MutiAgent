"""Existing pipeline modules wrapped as tools for the LLM controller.

Each tool reuses a node function from orchestration.py unchanged. A tool reads the
current plan from the injected graph state, writes its full result back to the state
via Command(update=...), and returns only a short summary to the LLM.
`state` and `tool_call_id` are injected by LangGraph and are not visible to the LLM.
"""

import json
from pathlib import Path
from typing import Annotated, Dict, Optional

from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.prebuilt import InjectedState
from langgraph.types import Command

from orchestration import agent1, agent2, agent3, agent_scope

TAXONOMY = set(json.loads((Path(__file__).resolve().parents[2] / "data" / "taxonomy_skills.json").read_text()))


def _reply(tool_call_id: str, summary: str, **update) -> Command:
    # Full result goes into the graph state; the LLM only sees the summary.
    return Command(update={**update, "messages": [ToolMessage(summary, tool_call_id=tool_call_id)]})


@tool
def analyze_skills(
    reason: str,
    state: Annotated[dict, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
    jd: Optional[str] = None,
) -> Command:
    """Re-analyze the job description and user background into skill weights.
    Use only when the target role or job description changes. Pass the new job
    description as `jd` if the user provided one."""
    jd = jd or state["jd"]
    update = agent1({**state, "jd": jd})
    top = sorted(update["weights"].items(), key=lambda x: -x[1])[:5]
    return _reply(tool_call_id, f"Re-weighted {len(update['weights'])} skills. Top 5: {top}", jd=jd, **update)


@tool
def retrieve_questions(
    skills: Dict[str, int],
    reason: str,
    state: Annotated[dict, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Retrieve extra candidate questions for some skills and add them to the
    candidate pool. `skills` maps taxonomy skill name -> number of questions needed.
    Does not change the plan itself."""
    unknown = [s for s in skills if s not in TAXONOMY]
    if unknown:
        return _reply(tool_call_id, f"Unknown skills {unknown}. Use taxonomy skill names, e.g. {sorted(state.get('weights') or {})[:5]}.")

    new = agent2({**state, "plan": skills})["retrieve_questions"]
    # Unlike agent2, append to the existing pool (deduplicated by id) instead of replacing it.
    pool = {k: list(v) for k, v in (state.get("retrieve_questions") or {}).items()}
    added = {}
    for skill, items in new.items():
        seen = {q["id"] for q in pool.get(skill, [])}
        fresh = [q for q in items if q["id"] not in seen]
        pool.setdefault(skill, []).extend(fresh)
        added[skill] = len(fresh)
    return _reply(tool_call_id, f"Added new candidates to pool: {added}", retrieve_questions=pool)


@tool
def generate_plan(
    reason: str,
    state: Annotated[dict, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
    days_left: Optional[int] = None,
) -> Command:
    """Rebuild the whole study plan from the current skill weights: re-plan scope,
    re-retrieve candidates, and re-schedule all days. Expensive; use only for a full
    replan (e.g. after analyze_skills for a new role), not for small adjustments."""
    current = {**state, "days_left": days_left or state["days_left"]}
    update = {"days_left": current["days_left"]}
    # Agent3 needs fresh candidates for the new quotas, so retrieval runs in between.
    for node in (agent_scope, agent2, agent3):
        result = node(current)
        current.update(result)
        update.update(result)
    summary = (
        f"New {len(update['days'])}-day plan with {update['total_questions']} questions. "
        f"Skill quotas: {update['plan']}"
    )
    return _reply(tool_call_id, summary, **update)


TOOLS = [analyze_skills, retrieve_questions, generate_plan]
