## weight_allocator.py: module C of skill_analyzer_agent - weight allocation to each level of preference

import math
# import json

class WeightAllocator:

    def __init__(self):
        # baseline importance by group (required > preferred > general)
        self.base_weights = {
            "required_mapped_skills": 3.0,
            "preferred_mapped_skills": 2.0,
            "general_mapped_skills": 1.0,
        }

        # sublinear repeat bonus strength (applied to repeats beyond the first)
        self.repeat_bonus = 0.5

        # small reward if a skill appears in multiple groups
        self.multi_group_bonus = 0.2

        # how strongly required dominates at the group-budget level
        # gamma=2 => budgets proportional to 3^2 : 2^2 : 1^2 = 9 : 4 : 1
        self.group_budget_gamma = 1.5

    def allocate_weights(self, mapped_skills: dict) -> dict:
        """
        Input example:
        {
          "required_mapped_skills": {"SQL": ["sql_basic", "sql_filter"]},
          "preferred_mapped_skills": {"Pandas": ["pandas_data_cleaning"]},
          "general_mapped_skills": {"data analysis": ["sql_basic", "pandas_data_cleaning"]}
        }

        Output: normalized weights that sum to ~1.0
        """

        skill_info = {}  # skill -> {"groups": set([...]), "count": float}

        # 1) collect group membership + fractional frequency (1/N per keyword)
        for group, kw_dict in (mapped_skills or {}).items():
            if group not in self.base_weights:
                continue

            for _, skills in (kw_dict or {}).items():
                skills = skills or []
                if not skills:
                    continue

                credit = 1.0 / len(skills)  # each keyword distributes 1.0 total credit

                for s in skills:
                    if s not in skill_info:
                        skill_info[s] = {"groups": set(), "count": 0.0}
                    skill_info[s]["groups"].add(group)
                    skill_info[s]["count"] += credit

        if not skill_info:
            return {}

        # helper: pick the strongest group as the primary group (required > preferred > general)
        def primary_group(groups: set[str]) -> str:
            return max(groups, key=lambda g: self.base_weights[g])

        # 2) compute per-skill raw score, and assign each skill to a primary group bucket
        group_to_skills = {g: [] for g in self.base_weights.keys()}
        raw_score = {}  # skill -> float

        for skill, info in skill_info.items():
            groups = info["groups"]
            count = info["count"]

            pg = primary_group(groups)
            baseline = self.base_weights[pg]

            # sublinear repeat boost: only kicks in after "1.0 total credit"
            repeat_credit = max(0.0, count - 1.0)
            repeat_boost = self.repeat_bonus * math.log1p(repeat_credit)

            # multi-group bonus
            group_boost = self.multi_group_bonus * max(0, len(groups) - 1)

            score = baseline + repeat_boost + group_boost
            raw_score[skill] = score
            group_to_skills[pg].append(skill)

        # 3) compute group budgets (required dominates regardless of how many general skills exist)
        # budgets ∝ (base_weight ** gamma), only among groups that actually have skills
        active_groups = [g for g, skills in group_to_skills.items() if skills]
        if not active_groups:
            return {}

        group_mass = {g: (self.base_weights[g] ** self.group_budget_gamma) for g in active_groups}
        mass_sum = sum(group_mass.values())
        group_budget = {g: (group_mass[g] / mass_sum) for g in active_groups}

        # 4) within each group, normalize scores and allocate that group's budget
        final_weights = {}

        for g in active_groups:
            skills = group_to_skills[g]
            ssum = sum(raw_score[s] for s in skills)

            # safety: if all scores are zero (shouldn't happen), split evenly
            if ssum <= 0:
                per = group_budget[g] / len(skills)
                for s in skills:
                    final_weights[s] = per
            else:
                for s in skills:
                    final_weights[s] = group_budget[g] * (raw_score[s] / ssum)

        # 5) final normalization (guard against tiny floating drift) + rounding
        total = sum(final_weights.values())
        if total <= 0:
            return {}

        return {s: round(w / total, 4) for s, w in final_weights.items()}

# ## Example test
# def main():

#     mapped_skills = {
#     "required_mapped_skills": {
#         "SQL": [
#         "sql_basic",
#         "sql_filter",
#         "sql_aggregation",
#         "sql_group_by",
#         "sql_sorting"
#         ],
#         "window functions": [
#         "sql_window_function"
#         ],
#         "complex analytical queries": []
#     },
#     "preferred_mapped_skills": {
#         "Pandas": [
#         "pandas_data_cleaning",
#         "pandas_data_manipulation",
#         "pandas_data_selection",
#         "pandas_data_reshaping",
#         "pandas_data_inspection"
#         ],
#         "random forest": [
#         "random_forest"
#         ]
#     },
#     "general_mapped_skills": {
#         "data analysis": [],
#         "metrics framework": [],
#         "statistical methods": [],
#         "A/B testing": [],
#         "causal inference": [],
#         "data analytics": [],
#         "business decisions": [],
#         "data science theories": [],
#         "methodology": [],
#         "analysis efficiency": [],
#         "data product tools": [
#         "pandas_data_cleaning",
#         "pandas_data_manipulation",
#         "pandas_data_selection"
#         ],
#         "Math": [
#         "algo_math"
#         ],
#         "Statistic": [],
#         "Data Science": [],
#         "Machine Learning": [
#         "supervised_machine learning",
#         "classification",
#         "clustering",
#         "decision_trees",
#         "neural_networks"
#         ],
#         "big data technologies": [
#         "databases"
#         ]
#     }
# }

#     allocator = WeightAllocator()
#     weights = allocator.allocate_weights(mapped_skills)

#     print(json.dumps(weights, indent=2))

# if __name__ == "__main__":
#     main()