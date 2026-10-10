import json
import math
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Dict, Any, Optional
# from openai import OpenAI
from scripts.langchain_llm import invoke_json
from scripts.Agent3.Planning_Agent import normalize_skill_name

MAX_SKILL_SHARE = 0.35  # no single skill takes more than about a third of the plan
QUESTIONS_PATH = Path(__file__).resolve().parents[1] / "data" / "questions_normalized.jsonl"


@lru_cache(maxsize=1)
def bank_supply() -> Counter:
    """How many questions the bank has per (normalized) skill."""
    supply = Counter()
    with QUESTIONS_PATH.open(encoding="utf-8") as f:
        for line in f:
            for skill in json.loads(line).get("taxonomy_skills") or []:
                supply[normalize_skill_name(skill)] += 1
    return supply


def cap_quotas(skill_plan: Dict[str, int], skill_weights: Dict[str, float], supply: Counter) -> Dict[str, int]:
    """Limit each quota to the questions the bank has for that skill and to MAX_SKILL_SHARE of the
    plan, then hand the excess to the other skills one question at a time (highest weight per
    already-assigned question first). If no skill can take more, the plan gets smaller."""
    total = sum(skill_plan.values())
    if not skill_plan or total <= 0:
        return skill_plan
    share_cap = max(math.ceil(MAX_SKILL_SHARE * total), math.ceil(total / len(skill_plan)))
    caps = {s: min(supply.get(normalize_skill_name(s), 0), share_cap) for s in skill_plan}
    plan = {s: min(q, caps[s]) for s, q in skill_plan.items()}
    for _ in range(total - sum(plan.values())):
        open_skills = [s for s in plan if plan[s] < caps[s]]
        if not open_skills:
            break
        best = max(open_skills, key=lambda s: skill_weights.get(s, 0) / (plan[s] + 1))
        plan[best] += 1
    return plan

## Problem
## theory：如何定义题目数量？
## 难度label如何呈现的逻辑仔细看下：retrieve时，结合difficulty distribution来retrieve对应skills的题目，如果没有这种combination或者不够了，按难度来retrieve足够数量的题目
def load_prompt(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


TEMPLATE_C = load_prompt("prompts/scope_planner.txt")


class ScopePlannerAgent:
    def __init__(self, client):
        self.client = client

    def infer_scope(self, days_left: int, user_desc: str, jd_text: str) -> Dict[str, Any]:
        prompt = f"""
            Days left:
            {days_left}

            User description:
            {user_desc}

            Job description:
            {jd_text}

            {TEMPLATE_C}
        """

        try:
            scope = invoke_json(
                prompt=prompt,
                temperature=0.3,
                client=self.client,
            )
        except Exception:
            # conservative fallback
            scope = {
                "total_questions": 40,
                "difficulty_distribution": {
                    "easy": 0.3,
                    "medium": 0.5,
                    "hard": 0.2,
                },
            }

        return scope

    def _extract_total_questions(self, scope: Dict[str, Any], days_left: int) -> int:
        """
        Robustly extract an integer total_questions from LLM output.
        If extraction fails, fall back to a policy based on days_left (not a magic constant).
        """
        tq = scope.get("total_questions")

        # case 1: already an int
        if isinstance(tq, int):
            return tq

        # case 2: numeric string
        if isinstance(tq, str):
            try:
                return int(tq)
            except ValueError:
                pass

        # case 3: dict wrapper (common LLM pattern)
        if isinstance(tq, dict):
            val = tq.get("value") or tq.get("count") or tq.get("total")
            if isinstance(val, int):
                return val
            if isinstance(val, str):
                try:
                    return int(val)
                except ValueError:
                    pass

        # final fallback: system policy (2–4 questions/day; pick 3 as default)
        return max(5, min(days_left * 3, 60))

    def allocate_by_weight(
        self,
        skill_weights: Dict[str, float],
        total_questions: int,
        min_per_skill: int = 1,
        max_skills: Optional[int] = None,
    ) -> Dict[str, int]:
        if not skill_weights or total_questions <= 0:
            return {}

        # keep only positive-weight skills
        items = [(s, w) for s, w in skill_weights.items() if w > 0]
        if not items:
            return {}

        # optional cap to avoid explosion
        if max_skills is not None and len(items) > max_skills:
            items.sort(key=lambda x: x[1], reverse=True)
            items = items[:max_skills]

        n = len(items)

        # If total_questions cannot cover min_per_skill for all, fallback: allocate 1 to top-K
        if total_questions < n * min_per_skill:
            items.sort(key=lambda x: x[1], reverse=True)
            allocated = {s: 0 for s, _ in items}
            for s, _ in items[:total_questions]:
                allocated[s] = 1
            return allocated

        # 1) give everyone the minimum
        allocated = {s: min_per_skill for s, _ in items}
        remaining = total_questions - n * min_per_skill

        if remaining == 0:
            return allocated

        # 2) distribute remaining proportionally by weights (largest remainder)
        weight_sum = sum(w for _, w in items)
        quotas = []
        for s, w in items:
            exact = remaining * (w / weight_sum)
            base = int(math.floor(exact))
            frac = exact - base
            quotas.append((s, base, frac))

        # add floors
        for s, base, _ in quotas:
            allocated[s] += base
        remainder = remaining - sum(base for _, base, _ in quotas)

        # add leftover by largest fractional parts
        quotas.sort(key=lambda x: x[2], reverse=True)
        for i in range(remainder):
            allocated[quotas[i][0]] += 1

        return allocated

    def run(
        self,
        days_left: int,
        user_desc: str,
        jd_text: str,
        skill_weights: Dict[str, float],
    ) -> Dict[str, Any]:
        scope = self.infer_scope(days_left, user_desc, jd_text)

        tq = self._extract_total_questions(scope, days_left)

        skill_plan = self.allocate_by_weight(
            skill_weights,
            tq
        )
        # A quota larger than the bank can supply used to be filled with unrelated questions.
        skill_plan = cap_quotas(skill_plan, skill_weights, bank_supply())

        return {
            "total_questions": sum(skill_plan.values()),
            "difficulty_distribution": scope.get(
                "difficulty_distribution",
                {"easy": 0.3, "medium": 0.5, "hard": 0.2},
            ),
            "skill_plan": skill_plan,
        }


# def main():
#     client = OpenAI()
#     jd = """
#     Responsibilities:
#     - Conduct data analysis in LIVE related business, including watching experience, creator ecosystem, agency management, algorithm improvement and etc..
#     - Design metrics framework to measure product healthiness, keep tracking of core metrics and understand root causes of metric movements.
#     - Conduct scientific evaluation with statistical methods, including A/B testing and casual inference.
#     - Identify growth opportunities with data analytics, and drive business decisions. Work with PM/MLE/RD to deliver product and strategy improvement.
#     - Research data science theories and methodology, improve analysis efficiency and data product tools.

#     Minimum Qualifications:
#     - Currently pursuing an Undergraduate/Master in Math, Statistic, Data Science, Machine Learning, or a related technical discipline.
#     - Expertise in SQL and programming in Python or R.
#     - Strong analytical and causal reasoning mindset, and rigidity on statistical correctness. Strong communication and passion about product challenges.

#     Preferred Qualifications:
#     - Experience of LIVE related business.
#     - Knowledge of machine learning and recommendation systems.
#     - Experience with big data technologies.
#     """
#     user = """
#     I really want to enhance my SQL coding ability, especially with window functions and more complex analytical queries. I also need help strengthening my Pandas coding skills, random forest related machine learning knowledge as well.
#     however, I do not have too much time for preparation.
#     """
#     days_left = 7
#     skill_weights = {
#         "sql_basic": 0.096,
#         "sql_filter": 0.096,
#         "sql_aggregation": 0.096,
#         "sql_group_by": 0.096,
#         "sql_sorting": 0.096,
#         "sql_window_function": 0.096,
#         "pandas_data_cleaning": 0.0547,
#         "pandas_data_manipulation": 0.0547,
#         "pandas_data_selection": 0.0547,
#         "pandas_data_reshaping": 0.0497,
#         "pandas_data_inspection": 0.0497,
#         "random_forest": 0.0497,
#         "algo_math": 0.0158,
#         "supervised_machine learning": 0.0158,
#         "classification": 0.0158,
#         "clustering": 0.0158,
#         "decision_trees": 0.0158,
#         "neural_networks": 0.0158,
#         "databases": 0.0158
#     }

#     planner = ScopePlannerAgent(client)
#     result = planner.run(
#         user_desc=user,
#         jd_text=jd,
#         skill_weights=skill_weights,
#         days_left=days_left
#     )

#     print(json.dumps(result, indent=2))


# if __name__ == "__main__":
#     main()
