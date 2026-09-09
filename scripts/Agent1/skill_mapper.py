## skill_mapper.py: module B of skill_analyzer_agent - technical skill keywords extraction from inputs

import json
from string import Template
from typing import List
# from openai import OpenAI
from scripts.langchain_llm import invoke_json

## Skill mapping
## 1) Separate functional/operational skills from difficulty labels
## 2) Ignore difficulty labels entirely (never output them)
## 3) Handle phrase types differently: FUNCTIONAL vs GENERIC vs COMBINED; drop overly generic phrases like data analysis, data scientist

## Problems:
## extracted words will be dropped if nothing is mapped
## when we need to map to algo

def load_prompt(path: str) -> Template:
    with open(path, "r", encoding="utf-8") as f:
        return Template(f.read())


def load_taxonomy(path: str) -> List[str]:
    with open(path, "r") as f:
        taxonomy = json.load(f)
    if not isinstance(taxonomy, list):
        raise ValueError("taxonomy_skills.json must contain a list of strings.")
    return taxonomy


SQL_BASE_PACK = [
    "sql_basic", 
    "sql_aggregation",
    "sql_filter", 
    "sql_group_by", 
    "sql_sorting"
]

PANDAS_BASE_PACK = [
    "pandas_data_cleaning",
    "pandas_data_selection",
    "pandas_data_reshaping",
    "pandas_data_manipulation",
    "pandas_data_inspection"
]

ALGO_BASE_PACK = [
    "algo_array_string",
    "algo_hash_map",
    "algo_sorting",
    "algo_dp"
]

TEMPLATE_B = load_prompt("prompts/skill_mapper.txt")
TAXONOMY_PATH = "data/taxonomy_skills.json"

class SkillMapper:
    """
    Module B: Convert extracted natural-language keywords to taxonomy skills
    using the LLM mapping prompt (TEMPLATE_B).
    """

    def __init__(self, client, taxonomy_path=TAXONOMY_PATH, template=TEMPLATE_B):
        self.client = client
        self.template = template
        self.taxonomy = load_taxonomy(taxonomy_path)

    # LLM single-skill mapping
    def _map_single(self, keyword: str) -> List[str]:
        prompt = self.template.substitute(
            KEYWORD=keyword,
            TAXONOMY=json.dumps(self.taxonomy),
            SQL_BASE_PACK=json.dumps(SQL_BASE_PACK),
            PANDAS_BASE_PACK=json.dumps(PANDAS_BASE_PACK),
            ALGO_BASE_PACK=json.dumps(ALGO_BASE_PACK),
        )

        try:
            parsed = invoke_json(
                prompt=prompt,
                temperature=0.0,
                client=self.client,
            )
            mapped = parsed.get("mapped_skills", [])

            # Only keep valid taxonomy skills
            return [s for s in mapped if s in self.taxonomy]

        except Exception as e:
            print("LLM mapping error:", e)
            return []

    # Batch mapping over all categories
    def map(self, extracted_keywords: dict) -> dict:
        """
        Input structure:
        {
            "required_keywords": [...],
            "preferred_keywords": [...],
            "general_keywords": [...]
        }

        Output structure:
        {
            "required_mapped_skills": { keyword: [taxonomy skills], ... },
            "preferred_mapped_skills": { ... },
            "general_mapped_skills": { ... }
        }
        """

        output = {}

        for category, kw_list in extracted_keywords.items():
            mapped_category = category.replace("keywords", "mapped_skills")
            output[mapped_category] = {}

            for kw in kw_list:
                mapped_skills = self._map_single(kw)
                output[mapped_category][kw] = mapped_skills

        return output


# ## Example test
# def main():
#     client = OpenAI()

#     extracted_keywords = {
#        "required_keywords": [
#             "SQL",
#             "window functions",
#             "complex analytical queries"
#         ],
#         "preferred_keywords": [
#             "Pandas",
#             "random forest"
#         ],
#         "general_keywords": [
#             "data analysis",
#             "metrics framework",
#             "statistical methods",
#             "A/B testing",
#             "causal inference",
#             "data analytics",
#             "business decisions",
#             "data science theories",
#             "methodology",
#             "analysis efficiency",
#             "data product tools",
#             "Math",
#             "Statistic",
#             "Data Science",
#             "Machine Learning",
#             "big data technologies"
#         ]
#     }

#     mapper = SkillMapper(client)
#     mapped = mapper.map(extracted_keywords)

#     print(json.dumps(mapped, indent=2))

# if __name__ == "__main__":
#     main()
