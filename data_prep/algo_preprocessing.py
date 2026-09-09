import json
import re
import os
from pathlib import Path
from bs4 import BeautifulSoup
from openai import OpenAI
from tqdm import tqdm

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "YOUR_API_KEY_HERE")
INPUT_FILE = Path("data/algorithms_raw.json")
OUTPUT_FILE = Path("data/algorithms.json")
TAXONOMY_MODEL = "gpt-5.1" 
SUMMARY_MODEL = "gpt-4o-mini"
ALGO_SKILL_TAXONOMY = [
    "algo_array_string", "algo_backtracking", "algo_beginner", "algo_binary_search",
    "algo_bit_manipulation", "algo_challenge", "algo_dp", "algo_easy", "algo_enumeration",
    "algo_graph", "algo_greedy", "algo_hard", "algo_hash_map", "algo_heap",
    "algo_intervals", "algo_linked_list", "algo_math", "algo_medium", "algo_prefix_sum",
    "algo_shortest_path", "algo_sliding_window", "algo_sorting", "algo_stack_queue",
    "algo_tree_bst", "algo_two_pointers",
]

client = OpenAI(api_key=OPENAI_API_KEY)

def clean_html(html_content):
    if not html_content:
        return ""
    soup = BeautifulSoup(html_content, "html.parser")
    text = soup.get_text(separator="\n", strip=True)
    text = re.sub(r'!\[.*?\]\(.*?\)', '', text)
    text = re.sub(r'\[([^\]]*)\]\([^)]+\)', r'\1', text)
    text = re.sub(r'https?://\S+', '', text)
    
    lines = text.split('\n')
    merged_lines = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if merged_lines and (line[0].islower() or not merged_lines[-1].endswith(('.', ':', '!', '?'))):
            merged_lines[-1] += " " + line
        else:
            merged_lines.append(line)
    text = "\n".join(merged_lines)
    headers = ["Example 1:", "Example 2:", "Example 3:", "Constraints:", "Input:", "Output:", "Explanation:"]
    for h in headers:
        text = text.replace(h, f"\n\n{h}")
    return re.sub(r'\n{3,}', '\n\n', text).strip()

def fix_scientific_notation(text):
    return re.sub(r'10\s+([0-9])', r'10^\1', text)

def remove_examples(text):
    text = re.sub(r"(?:\n|\r|\r\n)?Example[\s\S]*$", "", text)
    text = re.sub(r"\b([a-zA-Z])\s+(\d+)\b", r"\1_\2", text)
    text = re.sub(r"\b([a-zA-Z])\s+([a-zA-Z])\b", r"\1_\2", text)
    return text.strip()

def parse_nested_json(json_str, val):
    if not json_str:
        return val
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        return val

def parse_ac_rate(stats_json_str):
    stats = parse_nested_json(stats_json_str, {})
    raw_rate = stats.get('acRate')
    if not raw_rate:
        return None
    try:
        return round(float(raw_rate.strip('%')) / 100, 3)
    except ValueError:
        return None

def get_new_difficulty_tag(ac_rate):
    if ac_rate is None:
        return "algo_medium" 
    if ac_rate >= 0.70:
        return "algo_beginner"
    elif ac_rate >= 0.55:
        return "algo_easy"
    elif ac_rate >= 0.40:
        return "algo_medium"
    elif ac_rate >= 0.25:
        return "algo_hard"
    else:
        return "algo_challenge"

def remove_leetcode_headers(clean):
    lines = clean.splitlines()
    result = []
    started = False
    for line in lines:
        if re.match(r'\s*###\s+', line):
            started = True
        if started:
            result.append(line)
    return "\n".join(result).strip()

def extract_solution_summary(problem):
    sol_html = problem.get('solution')
    if sol_html:
        clean = clean_html(sol_html)
        clean = clean.replace("[TOC]", "")
        clean = remove_leetcode_headers(clean)
        clean = clean.strip()
        if len(clean) > 1: 
            return clean[:3000]
    return None

def map_topics_to_taxonomy(title, topics_list, description):
    taxonomy_json = json.dumps(ALGO_SKILL_TAXONOMY)
    prompt = f"""
    You are a classification model that maps algorithm problems into taxonomy skills.

    Taxonomy Skills (choose 1-4):
    {taxonomy_json}

    Rules:
    - Only return skills from the taxonomy.
    - Output must be a JSON list of strings.
    - No explanation.
    - No new labels.

    Problem:
    Title: {title}
    Topics: {topics_list}
    Description: {description}

    Return ONLY the JSON list.
    """

    response = client.chat.completions.create(
        model=TAXONOMY_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_completion_tokens=120
    )
    content = response.choices[0].message.content
    return json.loads(content)

def generate_solution_summary(title, topics, description):
    prompt = f"""
    You are an expert algorithm interview coach.
    
    Task: Provide a concise high-level solution summary for the following coding problem.
    
    Problem Title: {title}
    Topics: {topics}
    Problem Description: {description}
    Requirements:
    1. Summarize the **optimal approach** in 3-6 sentences.
    2. Explicitly mention the **Data Structures** and **Algorithms** used (e.g., "Use a Min-Heap to...", "Apply Sliding Window...").
    3. Mention Time and Space Complexity if possible.
    4. **DO NOT write code**. Pure text explanation only.
    
    Output format: Just the summary text.
    """
    response = client.chat.completions.create(
        model=SUMMARY_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_completion_tokens=150
    )
    return response.choices[0].message.content.strip()

def process_all_problems(raw_problems):
    processed_data = []
    problem_keep = {"Algorithms"}
    current_idx = 1    
    for p in tqdm(raw_problems):
        cat = p.get('category', 'Unknown')
        if cat not in problem_keep:
            continue
        p_id = p.get('frontendQuestionId')
        description_clean = fix_scientific_notation(clean_html(p.get('description')))
        if not p_id or not description_clean:
            continue
        title = p.get('title', 'Unknown')
        topics_list = p.get('topics', [])
        topics_str = ", ".join(topics_list)
        difficulty_raw = p.get('difficulty') 
        raw_content_emb = (
            f"Title: {title}\n"
            f"Difficulty: {difficulty_raw}\n"
            f"Topics: {topics_str}\n"
            f"Description: {description_clean}"
        )
        final_vector_content = remove_examples(raw_content_emb)
        sol_summary = extract_solution_summary(p)
        if not sol_summary:
            sol_summary = generate_solution_summary(title, topics_list, description_clean)
        llm_skills = map_topics_to_taxonomy(title, topics_list, description_clean)
        ac_rate = parse_ac_rate(p.get('stats'))
        new_difficulty_tag = get_new_difficulty_tag(ac_rate)
        final_taxonomy_skills = [new_difficulty_tag] + llm_skills
        metadata = {
            "id": f"algo_{current_idx}",
            "title": title,
            "category": cat,
            "taxonomy_skills": final_taxonomy_skills,
            "solution_summary": sol_summary,
            "url": p.get('url', ''),
        }
        processed_data.append({
            "vector_content": final_vector_content,
            "metadata": metadata
        })
        
        current_idx += 1
        
    return processed_data

if __name__ == "__main__":
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    final_result = process_all_problems(raw_data)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(final_result, f, indent=1, ensure_ascii=False)