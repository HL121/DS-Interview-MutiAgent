import json
import random
import re
import os
import argparse
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

load_dotenv()

RELEVANCE_WEIGHT = 5.0
DIRECT_SKILL_BONUS = 1.0
DIFFICULTY_NEED_BONUS = 1.0
DIFFICULTY_OVER_PENALTY = 0.5
QUESTION_OVERFLOW_PENALTY = 5.0
HARD_OVERFLOW_PENALTY = 4.0
SKILL_OVERFLOW_PENALTY = 2.0

class PlanSwap(BaseModel):
    from_day: int = Field(description="1-indexed source day for the task swap.")
    to_day: int = Field(description="1-indexed target day for the task swap.")
    task_id: str = Field(description="Task ID to move.")


class PlanReview(BaseModel):
    swap: List[PlanSwap] = Field(default_factory=list, description="Small task moves that improve cognitive flow.")
    notes: str = Field(default="", description="Optional explanation for the suggested swaps.")


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

def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    data = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def align_to_retrieval_format(raw_tasks):
    aligned = []
    for r in raw_tasks:
        meta = r.get("metadata", {}) or {}
        data = r.get("data", {}) or {}
        primary_url = meta.get("url") or meta.get("source_url")
        backup_url = data.get("backup_url")

        aligned.append({
            "score": None,
            "id": r.get("id"),
            "type": meta.get("type"),
            "title": meta.get("title"),
            "difficulty": meta.get("difficulty"),
            "taxonomy_skills": meta.get("taxonomy_skills", []),
            "url": primary_url,
            "backup_url": backup_url,
            "preview": r.get("vector_text", "")[:180].replace("\n", " "),
            "metadata": meta,
            "data": r.get("data", {})
        })
    return aligned

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


# USER PARSING  find strength and weakness of user in text
def parse_request(text: str) -> int:
    m = re.search(r"(\d+)\s*days?\b", text.lower())
    return int(m.group(1)) if m else 7


def build_user_profile(text: str):
    t = text.lower()
    strong, weak = set(), set()

    if "strong in" in t:
        part = t.split("strong in", 1)[1]
        part = part.split("weak in", 1)[0]
        strong |= {s.strip() for s in part.split(",") if s.strip()}

    if "weak in" in t:
        part = t.split("weak in", 1)[1]
        part = part.split("strong in", 1)[0]
        weak |= {s.strip() for s in part.split(",") if s.strip()}

    return {"strong": strong, "weak": weak}

def build_difficulty_curve(n):
    """
    Smooth difficulty curve:
    ramp up → peak → taper for review   The difficulty distribution we want
    """
    if n <= 0:
        return []

    raw = []
    for i in range(n):
        x = 0.0 if n == 1 else i / (n - 1)
        if x < 0.5:
            val = 0.6 + (x / 0.5) * 0.6
        else:
            val = 1.2 - ((x - 0.5) / 0.5) * 0.4
        raw.append(val)

    total = sum(raw)
    return [v / total for v in raw]


def adjusted_difficulty(task, profile):
    base = {"easy": 1, "medium": 2, "hard": 3}[task["difficulty"]]
    skills = [s.lower() for s in task.get("taxonomy_skills", [])]
    strong = [s.lower() for s in profile["strong"]]
    weak = [s.lower() for s in profile["weak"]]

    if any(w in skill for skill in skills for w in weak):
        return base * 1.3
    if any(st in skill for skill in skills for st in strong):
        return base * 0.7
    return base


def task_load(task):
    return float(task.get("planner_score", {"easy": 1, "medium": 2, "hard": 3}[task["difficulty"]]))

def rebalance_underfilled_days(days, day_scores, targets):
    """
    Move tasks from overloaded days to underfilled days.
    """
    num_days = len(days)

    for _ in range(2):  # two passes, same as Colab
        for i in range(num_days):
            if day_scores[i] >= targets[i]:
                continue

            # find a donor day
            donor = max(
                range(num_days),
                key=lambda j: day_scores[j] - targets[j]
            )

            if day_scores[donor] <= targets[donor]:
                continue
# Adjust the hardness of questions based on user's familarity with that topic / skill
            # move the hardest task
            hardest = max(
                days[donor],
                key=lambda t: {"easy": 1, "medium": 2, "hard": 3}[t["difficulty"]],
                default=None
            )

            if hardest:
                days[donor].remove(hardest)
                days[i].append(hardest)

                diff = task_load(hardest)
                day_scores[donor] -= diff
                day_scores[i] += diff

    return days, day_scores

def cap_overfilled_days(days, day_scores, targets):
    """
    Prevent any day from exceeding 1.3x its target difficulty.
    """
    num_days = len(days)
    if num_days <= 1:
        return days, day_scores

    for i in range(num_days):
        cap = 1.3 * targets[i]

        while day_scores[i] > cap and len(days[i]) > 1:
            # move easiest task
            easiest = min(
                days[i],
                key=lambda t: {"easy": 1, "medium": 2, "hard": 3}[t["difficulty"]]
            )

            # find best recipient
            recipient = min(
                [j for j in range(num_days) if j != i],
                key=lambda j: day_scores[j]
            )

            days[i].remove(easiest)
            days[recipient].append(easiest)

            diff = task_load(easiest)
            day_scores[i] -= diff
            day_scores[recipient] += diff

    return days, day_scores



#Skipping the reviewer and summary llm, return the plans generated by logic
def assign_tasks(tasks, targets, profile):
    scored = [
        {**t, "score": adjusted_difficulty(t, profile)}
        for t in tasks
    ]
    scored.sort(key=lambda x: -x["score"])

    num_days = len(targets)
    days = [[] for _ in range(num_days)]
    day_scores = [0.0] * num_days
    day_skills = [set() for _ in range(num_days)]

    MAX_SKILLS = 2   # Maximum number of questions should we focus on each day

    # seed one task per day
    for i in range(num_days):
        if not scored:
            break
        t = scored.pop(0)
        days[i].append(t)
        day_scores[i] += t["score"]
        if t["taxonomy_skills"]:
            day_skills[i].add(t["taxonomy_skills"][0])

    for task in scored:
        task_skill = task["taxonomy_skills"][0] if task["taxonomy_skills"] else "general"
        best_day, best_cost = None, float("inf")

        for i in range(num_days):

            skill_penalty = (
                3.0
                if task_skill not in day_skills[i]
                and len(day_skills[i]) >= MAX_SKILLS
                else 0.0
            )

            overload_penalty = max(0, day_scores[i] - targets[i])
            review_bonus = -1.0 if i >= num_days - 2 else 0.0

            cost = (
                abs((day_scores[i] + task["score"]) - targets[i])
                + skill_penalty
                + overload_penalty
                + review_bonus
            )

            if cost < best_cost:
                best_cost, best_day = cost, i

        if best_day is None:
            best_day = min(range(num_days), key=lambda i: len(days[i]))

        days[best_day].append(task)
        day_scores[best_day] += task["score"]
        day_skills[best_day].add(task_skill)

    return days, day_scores


def reorder(days):
    order = {"easy": 1, "medium": 2, "hard": 3}
    return [sorted(day, key=lambda t: order[t["difficulty"]]) for day in days]


def generate_study_plan_v2(tasks, user_text):
    days_n = parse_request(user_text)
    profile = build_user_profile(user_text)

    personalized = [adjusted_difficulty(t, profile) for t in tasks]
    total = sum(personalized)
    targets = [total * w for w in build_difficulty_curve(days_n)]

    days, day_scores = assign_tasks(tasks, targets, profile)
    days, day_scores = rebalance_underfilled_days(days, day_scores, targets)
    days, day_scores = cap_overfilled_days(days, day_scores, targets)

    return reorder(days)


@dataclass
class PlanConstraints:
    days_left: int
    total_questions: int
    skill_plan: Dict[str, int]
    difficulty_distribution: Dict[str, float]
    max_questions_per_day: int
    max_hard_per_day: int = 1
    max_skills_per_day: int = 2


def difficulty_value(difficulty: str) -> int:
    return {"easy": 1, "medium": 2, "hard": 3}.get(difficulty, 2)


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
    user_request: str,
    skill_plan: Optional[Dict[str, int]],
    difficulty_distribution: Optional[Dict[str, float]],
) -> PlanConstraints:
    days = days_left or parse_request(user_request)
    days = max(1, days)
    normalized_skill_plan = {normalize_skill_name(k): int(v) for k, v in (skill_plan or {}).items()}
    total_questions = sum(normalized_skill_plan.values()) or len(tasks)
    max_questions = max(1, math.ceil(total_questions / days) + 1)
    max_hard = 1 if days >= 3 else 2

    return PlanConstraints(
        days_left=days,
        total_questions=total_questions,
        skill_plan=normalized_skill_plan,
        difficulty_distribution=difficulty_distribution or {"easy": 0.3, "medium": 0.5, "hard": 0.2},
        max_questions_per_day=max_questions,
        max_hard_per_day=max_hard,
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


def task_priority(task: Dict[str, Any], profile: Dict[str, set]) -> float:
    relevance = relevance_score(task)
    return float(relevance) * RELEVANCE_WEIGHT + difficulty_value(task.get("difficulty", "medium"))


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


def select_final_tasks(
    tasks: List[Dict[str, Any]],
    constraints: PlanConstraints,
    user_desc: str,
) -> List[Dict[str, Any]]:
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


def daily_targets(tasks: List[Dict[str, Any]], constraints: PlanConstraints, profile: Dict[str, set]) -> List[float]:
    total = sum(adjusted_difficulty(task, profile) for task in tasks)
    if total <= 0:
        total = max(1, len(tasks))
    return [total * weight for weight in build_difficulty_curve(constraints.days_left)]


def day_skill_set(day: List[Dict[str, Any]]) -> set:
    return {task.get("requested_skill") or "general" for task in day}


def choose_best_day(
    task: Dict[str, Any],
    days: List[List[Dict[str, Any]]],
    day_scores: List[float],
    targets: List[float],
    constraints: PlanConstraints,
) -> int:
    best_day = 0
    best_cost = float("inf")
    task_score = float(task["planner_score"])
    task_skill = task.get("requested_skill") or "general"
    task_diff = task.get("difficulty", "medium")

    for idx, day in enumerate(days):
        skill_set = day_skill_set(day)
        hard_count = sum(1 for item in day if item.get("difficulty") == "hard")
        question_overflow = max(0, len(day) + 1 - constraints.max_questions_per_day)
        hard_overflow = 1 if task_diff == "hard" and hard_count >= constraints.max_hard_per_day else 0
        skill_overflow = 1 if task_skill not in skill_set and len(skill_set) >= constraints.max_skills_per_day else 0
        target_gap = abs((day_scores[idx] + task_score) - targets[idx])

        cost = (
            target_gap
            + question_overflow * QUESTION_OVERFLOW_PENALTY
            + hard_overflow * HARD_OVERFLOW_PENALTY
            + skill_overflow * SKILL_OVERFLOW_PENALTY
        )

        if cost < best_cost:
            best_cost = cost
            best_day = idx

    return best_day


def constraint_schedule_tasks(
    tasks: List[Dict[str, Any]],
    constraints: PlanConstraints,
    user_desc: str,
) -> List[List[Dict[str, Any]]]:
    profile = build_user_profile(user_desc)

    prepared = []
    seen_ids = set()
    for task in tasks:
        if task["id"] in seen_ids:
            continue
        seen_ids.add(task["id"])
        requested_skill = infer_requested_skill(task, constraints.skill_plan)
        prepared.append({
            **task,
            "requested_skill": requested_skill,
            "planner_score": adjusted_difficulty(task, profile),
            "priority": task_priority(task, profile),
        })

    prepared.sort(key=lambda item: item["priority"], reverse=True)

    days = [[] for _ in range(constraints.days_left)]
    day_scores = [0.0] * constraints.days_left
    targets = daily_targets(prepared, constraints, profile)

    for task in prepared:
        best_day = choose_best_day(task, days, day_scores, targets, constraints)
        days[best_day].append(task)
        day_scores[best_day] += task["planner_score"]

    days, day_scores = rebalance_underfilled_days(days, day_scores, targets)
    days, day_scores = cap_overfilled_days(days, day_scores, targets)
    return reorder(days)


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


# Using LLM reviewer to review, see any problem sequence to change
def safe_parse_json(text):
    """
    Safely parse JSON from LLM output.
    Returns None if parsing fails.
    """
    if not text or not text.strip():
        print("[LLM ERROR] Empty response")
        return None

    text = text.strip()

    # Remove markdown code fences if present
    text = re.sub(r"^```json", "", text)
    text = re.sub(r"^```", "", text)
    text = re.sub(r"```$", "", text)
    text = text.strip()

    # First attempt: parse directly
    try:
        return json.loads(text)
    except Exception:
        pass

    # Second attempt: extract first JSON object or array
    match = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", text)
    if match:
        try:
            return json.loads(match.group(1))
        except Exception as e:
            print("[LLM ERROR] JSON extraction failed")
            print("RAW:", repr(text))
            print("EXTRACTED:", match.group(1))
            return None

    print("[LLM ERROR] No JSON found")
    print("RAW OUTPUT:", repr(text))
    return None


def normalize_summaries(summaries):
    """
    Ensure summaries is always:
    List[{"day": int, "summary": str}]
    """
    if summaries is None:
        return None

    # Correct case
    if isinstance(summaries, list):
        if summaries and isinstance(summaries[0], dict):
            return summaries

    # Single object → wrap
    if isinstance(summaries, dict):
        if "day" in summaries and "summary" in summaries:
            return [summaries]

    # Anything else → reject
    print("[INFO] Invalid summaries format, skipping summaries")
    return None



def build_alignment_prompt():
    return ChatPromptTemplate.from_messages([
        (
            "system",
            "You are reviewing a data science interview preparation study plan generated by a deterministic algorithm. "
            "Only suggest small task moves if they improve cognitive flow. "
            "Do not add tasks, remove tasks, or reorder tasks within a day. "
            "Prefer moving hard tasks away from overloaded days. "
            "If the plan looks good, return an empty swap list.",
        ),
        ("human", "Input plan:\n{plan_json}"),
    ])

def review_plan_with_llm(days, client, day_metadata=None):
    metadata_by_day = {item["day"]: item for item in (day_metadata or [])}
    compact = [
        {
            "day": i + 1,
            "daily_goal": metadata_by_day.get(i + 1, {}).get("daily_goal"),
            "review_task": metadata_by_day.get(i + 1, {}).get("review_task"),
            "tasks": [
                {
                    "id": t["id"],
                    "difficulty": t["difficulty"],
                    "skills": t["taxonomy_skills"]
                }
                for t in day
            ],
        }
        for i, day in enumerate(days)
    ]

    llm = get_langchain_chat_model(client, temperature=0.1)
    chain = build_alignment_prompt() | llm.with_structured_output(PlanReview)
    review = chain.invoke({"plan_json": json.dumps(compact, indent=2)})
    return review.model_dump()


def apply_llm_swaps(days, result):
    if not result or "swap" not in result:
        return days

    id_map = {t["id"]: t for d in days for t in d}

    for s in result["swap"]:
        fd, td = s["from_day"] - 1, s["to_day"] - 1
        task = id_map.get(s["task_id"])
        if task and task in days[fd]:
            days[fd].remove(task)
            days[td].append(task)

    return days


def normalize_review(review):
    if not isinstance(review, dict):
        return None
    if "swap" not in review or not isinstance(review["swap"], list):
        return None
    return review


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
    user_request: str,
    days_left: Optional[int] = None,
    skill_plan: Optional[Dict[str, int]] = None,
    difficulty_distribution: Optional[Dict[str, float]] = None,
    jd_text: str = "",
    user_desc: str = "",
    use_llm: bool = True,
    client: Optional[Any] = None,
):
    """
    Unified entry point for Planning Agent (for orchestration).

    Parameters
    ----------
    tasks : List[Dict]
        Normalized tasks from Retrieval Agent.
    user_request : str
        Natural language user request.
    use_llm : bool
        Whether to use LLM reviewer + summary.
    client : Any | None
        Optional LangChain chat model. If omitted, ChatOpenAI is created from OPENAI_API_KEY.

    Returns
    -------
    days : List[List[Dict]]
        Study plan grouped by day.
    summaries : Optional[List[Dict]]
        Day-level summaries.
    """

    if skill_plan or difficulty_distribution or days_left:
        constraints = build_plan_constraints(
            tasks=tasks,
            days_left=days_left,
            user_request=user_request,
            skill_plan=skill_plan,
            difficulty_distribution=difficulty_distribution,
        )
        selected_tasks = select_final_tasks(
            tasks=tasks,
            constraints=constraints,
            user_desc=user_desc,
        )
        days = constraint_schedule_tasks(
            tasks=selected_tasks,
            constraints=constraints,
            user_desc=user_desc,
        )
    else:
        days = generate_study_plan_v2(tasks, user_request)

    day_metadata = summarize_day_metadata(days)

    summaries = deterministic_summaries(day_metadata)

    # V3 (LLM)
    if use_llm:
        review = normalize_review(review_plan_with_llm(days, client, day_metadata))
        if review:
            days = apply_llm_swaps(days, review)
            day_metadata = summarize_day_metadata(days)

        raw_summaries = summarize_days(days, client, day_metadata)
        summaries = normalize_summaries(raw_summaries) or summaries

    return days, summaries



# MAIN (CLI)
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/merged.jsonl")
    parser.add_argument("--sample", type=int, default=50)
    parser.add_argument("--no-llm", action="store_true")
    parser.add_argument(
        "--request",
        default="Create a 6 day plan. I am strong in SQL, weak in ML."
    )
    args = parser.parse_args()

    random.seed(42)

    # ---------- Load + sample data ----------
    raw = load_jsonl(Path(args.input))
    sampled = random.sample(raw, min(args.sample, len(raw)))

    # ---------- Build tasks ----------
    tasks = normalize_tasks(align_to_retrieval_format(sampled))

    # ---------- V2 planner ----------
    days = generate_study_plan_v2(tasks, args.request)

    # ---------- Optional V3 (LLM) ----------
    client = None
    if not args.no_llm:
        client = get_langchain_chat_model()

    days, summaries = run_planning_agent(
        tasks=tasks,
        user_request=args.request,
        use_llm=not args.no_llm,
        client=client
    )


    # ---------- Output ----------
    diff = {"easy": 1, "medium": 2, "hard": 3}

    summary_map = {}
    if summaries:
        summary_map = {s["day"]: s["summary"] for s in summaries}

    for i, day in enumerate(days, 1):
        print(f"\nDAY {i} — difficulty {sum(diff[t['difficulty']] for t in day)}")
        for t in day:
            print(f"  • {t['id']} — {t['title']} ({t['difficulty']})")
            if t.get("url"):
                print(f"      ↳ {t['url']}")

        if i in summary_map:
            print("Summary:", summary_map[i])


if __name__ == "__main__":
    main()




"""
Exampel Output (with LLM)

DAY 1 — difficulty 13
  • lc:algo_1825 — Minimum Hours of Training to Win a Competition (easy)
      ↳ https://leetcode.com/problems/minimum-hours-of-training-to-win-a-competition
  • lc:algo_2218 — Find Maximum Non-decreasing Array Length (hard)
      ↳ https://leetcode.com/problems/find-maximum-non-decreasing-array-length
  • lc:algo_1379 — Count Pairs With XOR in a Range (hard)
      ↳ https://leetcode.com/problems/count-pairs-with-xor-in-a-range
  • lc:algo_1542 — The Score of Students Solving Math Expression (hard)
      ↳ https://leetcode.com/problems/the-score-of-students-solving-math-expression
  • lc:algo_382 — Smallest Good Base (hard)
      ↳ https://leetcode.com/problems/smallest-good-base
Summary: Great start with a mix of easy and hard problems! Focus on 'Minimum Hours of Training to Win a Competition' (algo_2605) to build your foundational skills in problem-solving. Tackling harder questions like 'Count Pairs With XOR in a Range' will enhance your analytical thinking, which is crucial for technical interviews.

DAY 2 — difficulty 18
  • lc:algo_94 — Binary Tree Inorder Traversal (easy)
      ↳ https://leetcode.com/problems/binary-tree-inorder-traversal
  • lc:algo_1364 — Check if Binary String Has at Most One Segment of Ones (easy)
      ↳ https://leetcode.com/problems/check-if-binary-string-has-at-most-one-segment-of-ones
  • lc:algo_369 — Validate IP Address (medium)
      ↳ https://leetcode.com/problems/validate-ip-address
  • lc:algo_2284 — Maximum Palindromes After Operations (medium)
      ↳ https://leetcode.com/problems/maximum-palindromes-after-operations
  • lc:algo_1704 — Find Players With Zero or One Losses (medium)
      ↳ https://leetcode.com/problems/find-players-with-zero-or-one-losses
  • lc:algo_888 — Longest Well-Performing Interval (medium)
      ↳ https://leetcode.com/problems/longest-well-performing-interval
  • lc:algo_2399 — Find the Minimum Area to Cover All Ones I (medium)
      ↳ https://leetcode.com/problems/find-the-minimum-area-to-cover-all-ones-i
  • lc:algo_1125 — Maximum Area of a Piece of Cake After Horizontal and Vertical Cuts (medium)
      ↳ https://leetcode.com/problems/maximum-area-of-a-piece-of-cake-after-horizontal-and-vertical-cuts
  • lc:algo_12 — Integer to Roman (medium)
      ↳ https://leetcode.com/problems/integer-to-roman
  • lc:algo_639 — Masking Personal Information (medium)
      ↳ https://leetcode.com/problems/masking-personal-information
Summary: You're making excellent progress! Begin with 'Binary Tree Inorder Traversal' (algo_2606) to strengthen your understanding of tree traversal techniques. As you move to medium problems like 'Longest Well-Performing Interval', you'll practice critical thinking and time complexity analysis, both vital for coding interviews.

DAY 3 — difficulty 12
  • th:theory_00057 — What is feature selection? Why do we need it? (easy)
  • lc:algo_163 — Majority Element (easy)
      ↳ https://leetcode.com/problems/majority-element
  • lc:algo_108 — Convert Sorted Array to Binary Search Tree (easy)
      ↳ https://leetcode.com/problems/convert-sorted-array-to-binary-search-tree
  • lc:algo_938 — Count Vowels Permutation (hard)
      ↳ https://leetcode.com/problems/count-vowels-permutation
  • lc:algo_800 — Subarrays with K Different Integers (hard)
      ↳ https://leetcode.com/problems/subarrays-with-k-different-integers
  • lc:algo_2647 — Shortest Path in a Weighted Tree (hard)
      ↳ https://leetcode.com/problems/shortest-path-in-a-weighted-tree
Summary: Keep up the momentum! Start with 'What is feature selection? Why do we need it?' to grasp essential concepts in data science. As you tackle 'Count Vowels Permutation', you'll enhance your combinatorial problem-solving skills, which are often tested in interviews.

DAY 4 — difficulty 12
  • lc:algo_365 — Island Perimeter (easy)
      ↳ https://leetcode.com/problems/island-perimeter
  • th:theory_00157 — What are good baselines when building a recommender system? (easy)
  • th:theory_00072 — How do we know how many trees we need in random forest? (easy)
  • lc:algo_2451 — Final Array State After K Multiplication Operations II (hard)
      ↳ https://leetcode.com/problems/final-array-state-after-k-multiplication-operations-ii
  • lc:algo_881 — Parsing A Boolean Expression (hard)
      ↳ https://leetcode.com/problems/parsing-a-boolean-expression
  • lc:algo_557 — Parse Lisp Expression (hard)
      ↳ https://leetcode.com/problems/parse-lisp-expression
Summary: You're doing fantastic! Begin with 'Island Perimeter' to solidify your understanding of grid-based problems. As you progress to 'Final Array State After K Multiplication Operations II', you'll sharpen your skills in algorithm design and optimization, which are crucial for technical interviews.

DAY 5 — difficulty 22
  • lc:algo_2605 — Check If Digits Are Equal in String After Operations I (easy)
      ↳ https://leetcode.com/problems/check-if-digits-are-equal-in-string-after-operations-i
  • lc:algo_88 — Merge Sorted Array (easy)
      ↳ https://leetcode.com/problems/merge-sorted-array
  • lc:algo_1112 — Number of Students Doing Homework at a Given Time (easy)
      ↳ https://leetcode.com/problems/number-of-students-doing-homework-at-a-given-time
  • lc:algo_2219 — Matrix Similarity After Cyclic Shifts (easy)
      ↳ https://leetcode.com/problems/matrix-similarity-after-cyclic-shifts
  • lc:algo_2055 — Minimum String Length After Removing Substrings (easy)
      ↳ https://leetcode.com/problems/minimum-string-length-after-removing-substrings
  • lc:algo_1124 — Maximum Product of Two Elements in an Array (easy)
      ↳ https://leetcode.com/problems/maximum-product-of-two-elements-in-an-array
  • lc:algo_622 — Binary Tree Pruning (medium)
      ↳ https://leetcode.com/problems/binary-tree-pruning
  • lc:algo_867 — Letter Tile Possibilities (medium)
      ↳ https://leetcode.com/problems/letter-tile-possibilities
  • lc:algo_404 — Find Largest Value in Each Tree Row (medium)
      ↳ https://leetcode.com/problems/find-largest-value-in-each-tree-row
  • lc:algo_1456 — Count Sub Islands (medium)
      ↳ https://leetcode.com/problems/count-sub-islands
  • lc:algo_1394 — Minimum Sideway Jumps (medium)
      ↳ https://leetcode.com/problems/minimum-sideway-jumps
  • lc:algo_2458 — K-th Nearest Obstacle Queries (medium)
      ↳ https://leetcode.com/problems/k-th-nearest-obstacle-queries
  • lc:algo_1069 — Count Number of Teams (medium)
      ↳ https://leetcode.com/problems/count-number-of-teams
  • lc:SQL_82 — Investments In 2016 (medium)
      ↳ https://leetcode.com/problems/investments-in-2016/
Summary: Great job on reaching day 5! Start with 'Check If Digits Are Equal in String After Operations I' to reinforce your string manipulation skills. Moving on to 'Binary Tree Pruning' will help you practice tree algorithms, a common topic in coding interviews.

DAY 6 — difficulty 17
  • lc:algo_1717 — Calculate Digit Sum of a String (easy)
      ↳ https://leetcode.com/problems/calculate-digit-sum-of-a-string
  • lc:algo_442 — Array Nesting (medium)
      ↳ https://leetcode.com/problems/array-nesting
  • lc:algo_989 — Sum of Mutated Array Closest to Target (medium)
      ↳ https://leetcode.com/problems/sum-of-mutated-array-closest-to-target
  • lc:algo_900 — Binary Tree Coloring Game (medium)
      ↳ https://leetcode.com/problems/binary-tree-coloring-game
  • lc:algo_405 — Longest Palindromic Subsequence (medium)
      ↳ https://leetcode.com/problems/longest-palindromic-subsequence
  • lc:algo_342 — Find All Anagrams in a String (medium)
      ↳ https://leetcode.com/problems/find-all-anagrams-in-a-string
  • lc:algo_2404 — Find the Maximum Length of Valid Subsequence II (medium)
      ↳ https://leetcode.com/problems/find-the-maximum-length-of-valid-subsequence-ii
  • lc:algo_1714 — Number of Ways to Buy Pens and Pencils (medium)
      ↳ https://leetcode.com/problems/number-of-ways-to-buy-pens-and-pencils
  • lc:algo_116 — Populating Next Right Pointers in Each Node (medium)
      ↳ https://leetcode.com/problems/populating-next-right-pointers-in-each-node
Summary: You're almost there! Begin with 'Calculate Digit Sum of a String' to warm up your problem-solving skills. As you tackle 'Longest Palindromic Subsequence', you'll enhance your dynamic programming abilities, which are essential for many technical interviews.

"""
