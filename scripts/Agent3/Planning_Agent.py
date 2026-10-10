"""Agent 3: select the final questions from the candidate pool, then schedule them across days.

Selection: greedy per skill quota, preferring relevant candidates and filling the target
difficulty mix. Scheduling: schedule() with three rules (balanced daily counts with a lighter
review day; at most one hard question per day, placed toward the end; easy -> medium ramp).
The LLM is only used to write the daily summaries.
"""

import json
import math
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

load_dotenv()

RELEVANCE_WEIGHT = 5.0
DIRECT_SKILL_BONUS = 1.0
DIFFICULTY_NEED_BONUS = 1.0
DIFFICULTY_OVER_PENALTY = 0.5
DIFF = {"easy": 1, "medium": 2, "hard": 3}


class DailySummary(BaseModel):
    day: int = Field(description="1-indexed study day.")
    summary: str = Field(description="Encouraging, actionable daily study summary.")


class DailySummaryList(BaseModel):
    summaries: List[DailySummary] = Field(description="Daily summaries for the full study plan.")


def get_langchain_chat_model(client=None, temperature: float = 0.1):
    if client is not None and hasattr(client, "with_structured_output"):
        return client

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")
    return ChatOpenAI(model="gpt-4o-mini", temperature=temperature, api_key=api_key)


def normalize_tasks(raw_tasks):
    normalized = []
    for r in raw_tasks:
        meta = r.get("metadata", {}) or {}
        data = r.get("data", {}) or {}

        task_id = r.get("id")
        title = r.get("title") or meta.get("title")
        difficulty = r.get("difficulty") or meta.get("difficulty")

        if not task_id or not title or not difficulty:
            continue

        skills = r.get("taxonomy_skills") or meta.get("taxonomy_skills", []) or []
        if isinstance(skills, str):
            skills = [skills]

        normalized.append({
            "id": task_id,
            "type": r.get("type") or meta.get("type"),
            "category": r.get("category") or meta.get("category"),
            "title": title,
            "difficulty": difficulty,
            "taxonomy_skills": skills,
            "url": r.get("url") or meta.get("url"),
            "backup_url": r.get("backup_url") or data.get("backup_url"),
            "retrieval_score": r.get("retrieval_score") or r.get("score"),
            "adjusted_score": r.get("adjusted_score"),
            "selection_reason": r.get("selection_reason") or data.get("selection_reason"),
            "requested_skill": r.get("requested_skill") or meta.get("requested_skill"),
            "requested_quota": r.get("requested_quota") or meta.get("requested_quota"),
        })
    return normalized


# ---------- selection ----------
@dataclass
class PlanConstraints:
    days_left: int
    total_questions: int
    skill_plan: Dict[str, int]
    difficulty_distribution: Dict[str, float]


def difficulty_value(difficulty: str) -> int:
    return DIFF.get(difficulty, 2)


def normalize_skill_name(skill: str) -> str:
    skill = (skill or "").replace("\xa0", " ").lower().strip()
    skill = re.sub(r"[^a-z0-9_]+", "_", skill)
    return re.sub(r"_+", "_", skill).strip("_")


def infer_requested_skill(task: Dict[str, Any], skill_plan: Dict[str, int]) -> str:
    if task.get("requested_skill"):
        return normalize_skill_name(task["requested_skill"])

    task_skills = {normalize_skill_name(s) for s in task.get("taxonomy_skills", [])}
    for skill in skill_plan:
        normalized = normalize_skill_name(skill)
        if normalized in task_skills:
            return normalized
    return next(iter(task_skills), "general")


def build_plan_constraints(
    tasks: List[Dict[str, Any]],
    days_left: Optional[int],
    skill_plan: Optional[Dict[str, int]],
    difficulty_distribution: Optional[Dict[str, float]],
) -> PlanConstraints:
    normalized_skill_plan = {normalize_skill_name(k): int(v) for k, v in (skill_plan or {}).items()}
    return PlanConstraints(
        days_left=max(1, days_left or 7),
        total_questions=sum(normalized_skill_plan.values()) or len(tasks),
        skill_plan=normalized_skill_plan,
        difficulty_distribution=difficulty_distribution or {"easy": 0.3, "medium": 0.5, "hard": 0.2},
    )


def normalized_distribution(distribution: Dict[str, float]) -> Dict[str, float]:
    cleaned = {k: max(0.0, float(v)) for k, v in distribution.items() if k in {"easy", "medium", "hard"}}
    total = sum(cleaned.values())
    if total <= 0:
        return {"easy": 0.3, "medium": 0.5, "hard": 0.2}
    return {k: cleaned.get(k, 0.0) / total for k in ("easy", "medium", "hard")}


def target_difficulty_counts(total_questions: int, distribution: Dict[str, float]) -> Dict[str, int]:
    distribution = normalized_distribution(distribution)
    exact = {
        difficulty: total_questions * distribution[difficulty]
        for difficulty in ("easy", "medium", "hard")
    }
    counts = {difficulty: int(math.floor(value)) for difficulty, value in exact.items()}
    remainder = total_questions - sum(counts.values())
    ranked = sorted(
        exact,
        key=lambda difficulty: exact[difficulty] - counts[difficulty],
        reverse=True,
    )
    for difficulty in ranked[:remainder]:
        counts[difficulty] += 1
    return counts


def relevance_score(task: Dict[str, Any]) -> float:
    score = task.get("adjusted_score")
    if score is None:
        score = task.get("retrieval_score")
    if score is None:
        score = task.get("score")
    return float(score or 0.0)


def candidate_selection_score(
    task: Dict[str, Any],
    requested_skill: str,
    selected_difficulty_counts: Counter,
    target_counts: Dict[str, int],
) -> float:
    difficulty = task.get("difficulty", "medium")
    task_skills = {normalize_skill_name(skill) for skill in task.get("taxonomy_skills", [])}
    score = relevance_score(task) * RELEVANCE_WEIGHT
    score += DIRECT_SKILL_BONUS if requested_skill in task_skills else 0.0
    score += (
        DIFFICULTY_NEED_BONUS
        if selected_difficulty_counts[difficulty] < target_counts.get(difficulty, 0)
        else -DIFFICULTY_OVER_PENALTY
    )
    return score


def group_candidates_by_skill(
    tasks: List[Dict[str, Any]],
    constraints: PlanConstraints,
) -> Dict[str, List[Dict[str, Any]]]:
    grouped = defaultdict(list)
    for task in tasks:
        requested_skill = infer_requested_skill(task, constraints.skill_plan)
        grouped[requested_skill].append({**task, "requested_skill": requested_skill})
    for skill in grouped:
        grouped[skill].sort(key=relevance_score, reverse=True)
    return grouped


def select_final_tasks(tasks: List[Dict[str, Any]], constraints: PlanConstraints) -> List[Dict[str, Any]]:
    grouped = group_candidates_by_skill(tasks, constraints)
    target_counts = target_difficulty_counts(constraints.total_questions, constraints.difficulty_distribution)
    selected_difficulty_counts = Counter()
    selected = []
    selected_ids = set()

    for skill, quota in constraints.skill_plan.items():
        candidates = grouped.get(skill, [])
        for _ in range(max(0, quota)):
            available = [candidate for candidate in candidates if candidate["id"] not in selected_ids]
            if not available:
                break
            best = max(
                available,
                key=lambda task: candidate_selection_score(
                    task,
                    skill,
                    selected_difficulty_counts,
                    target_counts,
                ),
            )
            selected.append(best)
            selected_ids.add(best["id"])
            selected_difficulty_counts[best.get("difficulty", "medium")] += 1

    needed = constraints.total_questions - len(selected)
    if needed > 0:
        remaining = []
        for candidates in grouped.values():
            remaining.extend(candidate for candidate in candidates if candidate["id"] not in selected_ids)
        remaining.sort(
            key=lambda task: candidate_selection_score(
                task,
                task.get("requested_skill", "general"),
                selected_difficulty_counts,
                target_counts,
            ),
            reverse=True,
        )
        for task in remaining[:needed]:
            selected.append(task)
            selected_ids.add(task["id"])
            selected_difficulty_counts[task.get("difficulty", "medium")] += 1

    return selected


# ---------- scheduling ----------
def schedule(tasks: List[dict], n_days: int, per_day: Optional[int] = None) -> List[List[dict]]:
    """Balanced counts, at most one hard question per day placed toward the end,
    easy -> medium order across days, and a lighter last day for review."""
    n = len(tasks)
    weights = [1.0] * n_days
    if n_days > 1:
        weights[-1] = 0.5
    raw = [n * w / sum(weights) for w in weights]
    targets = [int(x) for x in raw]
    for i in sorted(range(n_days), key=lambda i: raw[i] - targets[i], reverse=True)[: n - sum(targets)]:
        targets[i] += 1
    if per_day:
        targets = [min(t, per_day) for t in targets]

    days = [[] for _ in range(n_days)]
    hard = [t for t in tasks if t.get("difficulty") == "hard"]
    rest = sorted(
        (t for t in tasks if t.get("difficulty") != "hard"),
        key=lambda t: (DIFF.get(t.get("difficulty"), 2), t.get("requested_skill") or ""),
    )
    # Hard questions: one per day, starting from the day before the review day.
    hard_days = list(range(n_days - 2, -1, -1)) if n_days > 1 else [0]
    for i, task in enumerate(hard):
        days[hard_days[i % len(hard_days)]].append(task)
    # Others fill the earliest day that still has room, so difficulty ramps up over the days.
    for task in rest:
        open_days = [i for i in range(n_days) if len(days[i]) < targets[i]]
        days[open_days[0] if open_days else min(range(n_days), key=lambda i: len(days[i]))].append(task)
    return [sorted(day, key=lambda t: DIFF.get(t.get("difficulty"), 2)) for day in days]


# ---------- daily summaries ----------
def summarize_day_metadata(days: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    summaries = []
    for idx, day in enumerate(days, start=1):
        difficulty_mix = Counter(task.get("difficulty", "medium") for task in day)
        type_mix = Counter(task.get("type", "unknown") for task in day)
        skills = []
        seen = set()
        for task in day:
            skill = task.get("requested_skill") or (task.get("taxonomy_skills") or ["general"])[0]
            if skill not in seen:
                skills.append(skill)
                seen.add(skill)

        hard_count = difficulty_mix.get("hard", 0)
        daily_goal = "Practice " + ", ".join(skills[:2]) if skills else "Practice core interview skills"
        if hard_count:
            daily_goal += f" with {hard_count} hard challenge{'s' if hard_count > 1 else ''}"

        review_task = ""
        if idx == len(days):
            review_task = "Review mistakes, revisit weak skills, and summarize reusable patterns."
        elif hard_count:
            review_task = "After hard questions, write down the key pattern and one common mistake to avoid."

        summaries.append({
            "day": idx,
            "daily_goal": daily_goal,
            "estimated_time_minutes": sum(30 + 20 * (difficulty_value(task.get("difficulty", "medium")) - 1) for task in day),
            "skill_coverage": skills,
            "difficulty_mix": dict(difficulty_mix),
            "type_mix": dict(type_mix),
            "review_task": review_task,
        })
    return summaries


def deterministic_summaries(day_metadata: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    summaries = []
    for item in day_metadata:
        skills = ", ".join(item.get("skill_coverage", [])[:3]) or "core interview skills"
        difficulty_mix = item.get("difficulty_mix", {})
        mix_text = ", ".join(f"{count} {difficulty}" for difficulty, count in difficulty_mix.items())
        review = item.get("review_task")
        summary = (
            f"{item['daily_goal']}. Focus on {skills}; today's mix is {mix_text or 'balanced practice'} "
            f"with about {item['estimated_time_minutes']} minutes of work."
        )
        if review:
            summary += f" {review}"
        summaries.append({"day": item["day"], "summary": summary})
    return summaries


def normalize_summaries(summaries):
    """Ensure summaries is a list of {"day": int, "summary": str}; otherwise None."""
    if isinstance(summaries, list) and summaries and isinstance(summaries[0], dict):
        return summaries
    if isinstance(summaries, dict) and "day" in summaries and "summary" in summaries:
        return [summaries]
    return None


def build_summary_prompt():
    return ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a supportive study coach helping with interview prep. "
            "Do not change tasks or days. "
            "Write 2-3 encouraging, actionable sentences per day. "
            "You may reference question titles or IDs to help users locate tasks. "
            "Explain what skills the user is practicing and why it matters.",
        ),
        ("human", "Input plan:\n{plan_json}"),
    ])


def summarize_days(days, client, day_metadata=None):
    metadata_by_day = {item["day"]: item for item in (day_metadata or [])}
    compact = [
        {
            "day": i + 1,
            "daily_goal": metadata_by_day.get(i + 1, {}).get("daily_goal"),
            "skill_coverage": metadata_by_day.get(i + 1, {}).get("skill_coverage", []),
            "difficulty_mix": metadata_by_day.get(i + 1, {}).get("difficulty_mix", {}),
            "review_task": metadata_by_day.get(i + 1, {}).get("review_task", ""),
            "tasks": [
                {"title": t["title"], "difficulty": t["difficulty"]}
                for t in day
            ],
        }
        for i, day in enumerate(days)
    ]

    llm = get_langchain_chat_model(client, temperature=0.3)
    chain = build_summary_prompt() | llm.with_structured_output(DailySummaryList)
    result = chain.invoke({"plan_json": json.dumps(compact, indent=2)})
    return [summary.model_dump() for summary in result.summaries]


def run_planning_agent(
    tasks: List[Dict[str, Any]],
    user_request: str = "",
    days_left: Optional[int] = None,
    skill_plan: Optional[Dict[str, int]] = None,
    difficulty_distribution: Optional[Dict[str, float]] = None,
    jd_text: str = "",
    user_desc: str = "",
    use_llm: bool = True,
    client: Optional[Any] = None,
):
    """Select questions, schedule them, and write daily summaries (LLM only for the wording).

    Returns (days, summaries): the plan grouped by day, and [{"day", "summary"}] per day.
    """
    constraints = build_plan_constraints(tasks, days_left, skill_plan, difficulty_distribution)
    selected = select_final_tasks(tasks, constraints)
    days = schedule(selected, constraints.days_left)

    day_metadata = summarize_day_metadata(days)
    summaries = deterministic_summaries(day_metadata)
    if use_llm:
        summaries = normalize_summaries(summarize_days(days, client, day_metadata)) or summaries
    return days, summaries
