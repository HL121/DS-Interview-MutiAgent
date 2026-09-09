import json
import re
import os
from pathlib import Path
from openai import OpenAI
from tqdm import tqdm

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "YOUR_API_KEY_HERE")

INPUT_FILE = Path("data/leetcode/leetcode_problems_raw.json")
OUTPUT_FILE = Path("data/leetcode/pandas.json")
MODEL_NAME = "gpt-4o"
VALID_PANDAS_SKILLS = [
    "pandas_data_inspection",
    "pandas_data_selection",
    "pandas_data_cleaning",
    "pandas_data_manipulation",
    "pandas_data_reshaping"
]

client = OpenAI(api_key=OPENAI_API_KEY)
def generate_clean_description(title, difficulty, description):
    system_prompt = (
        "You are an expert Data Scientist and Technical Communicator. "
        "Your role is to distill coding problem descriptions into clear, concise, logic-focused text. "
        "You serve as a pre-processor to remove noise for a user."
    )
    user_prompt = f"""
    Please summarize the following LeetCode problem description into a clean, text-only paragraph.

    *** STRICT RULES ***:
    1. **NO TABLES**: Do not output any Markdown tables, ASCII tables, or structured grids.
    2. **NO HTML**: Remove all HTML tags completely.
    3. **LOGIC ONLY**: Instead of showing example data (like "Alice", "Bob"), describe the *structure* and the *task*. 
    4. **Concise**: Keep it short and direct.

    *** Problem Input ***
    - Title: {title}
    - Difficulty: {difficulty}
    - Raw Description: 
    {description}
    """
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
        max_completion_tokens=150
    )
    return response.choices[0].message.content.strip()

def generate_pandas_tags(title, description, solution):
    system_prompt = f"""
    You are an expert Taxonomy Specialist for Data Science education.
    Your task is to tag LeetCode Pandas problems using a controlled vocabulary.
    
    *** CONTROLLED VOCABULARY ***
    {json.dumps(VALID_PANDAS_SKILLS)}

    *** RULE ***
    You MUST select from the above list ONLY.
    """
    user_prompt = f"""
    Analyze the following problem and assign **AT LEAST TWO (2)** tags.
    
    Input Data:
    - Title: {title}
    - Description: {description}
    - Solution Logic: {solution}

    Output Format:
    Return ONLY a JSON array of strings. Example: ["pandas_data_selection", "pandas_data_manipulation"]
    """
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": system_prompt},{"role": "user", "content": user_prompt}],
        max_completion_tokens=100
    )
    content = response.choices[0].message.content.strip()
    return json.loads(content)

def clean_solution(raw_text):
    prompt = f"""
You are a professional data scientist. Your task is to rewrite the following solution explanation into a short, plain-text summary.

Strict rules:
- Do not include any code or code blocks.
- Do not include HTML tags, markdown, symbols, lists, tables, or links.
- Do not mention images or figures.
- Do not explain pandas parameters or functions in detail.
- Output must be simple English sentences only.
- Output must be 4 to 6 concise sentences.
- Focus only on the high-level idea of the solution.

Rewrite the following text into a plain-text summary:

    {raw_text}
    """

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}]
    )
    content = response.choices[0].message.content.strip()
    return content

def process_pandas_problems(raw_data):
    processed_data = []
    pandas_problems = [p for p in raw_data if p.get("category") == "pandas"]
    for idx, item in enumerate(tqdm(pandas_problems), start=1):
        description_raw = item.get('description', '')
        title = item.get('title', 'Unknown')
        difficulty = item.get("difficulty", "Medium")
        raw_solution = item.get("solution", "")
        pre_cleaned = re.sub(r'\[TOC\]\s+## Solution\s+---\s+### Overview\s+', '', raw_solution)
        final_solution_summary = clean_solution(pre_cleaned)
        description_clean = generate_clean_description(title, difficulty, description_raw)
        skill_tags = generate_pandas_tags(title, description_clean, final_solution_summary)
        content_emb = (
            f"Title: {title}\n"
            f"Difficulty: {difficulty}\n"
            f"Description: {description_clean}"
        )
        metadata = {
            "id": f"pandas_{idx}",
            "title": title,
            "category": "Pandas",
            "taxonomy_skills": ["pandas_easy"] + skill_tags,
            "solution_summary": final_solution_summary,
            "url": item.get('url'),
        }
        processed_data.append({
            "vector_content": content_emb,
            "metadata": metadata
        })
    return processed_data

if __name__ == "__main__":
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    pandas_result = process_pandas_problems(raw_data)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(pandas_result, f, indent=1, ensure_ascii=False)