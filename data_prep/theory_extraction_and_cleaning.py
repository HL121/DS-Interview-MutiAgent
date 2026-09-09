"""
Theory QA Markdown Processor

- Extracts Q&A from markdown files
- Cleans and normalizes text
- Repairs taxonomy/subdomains
- Outputs JSONL for DS Interview Agent ingestion
"""

import json
import re
from pathlib import Path
from tqdm import tqdm
from collections import Counter


# CONFIG 
MD_DIR = Path("data/TheoryQA")          # folder containing .md files
OUTPUT_PATH = Path("data/Theory.jsonl")


# BASIC CLEANING
emoji_pattern = re.compile(r"[\U00010000-\U0010ffff]", flags=re.UNICODE)

def initial_clean(s: str) -> str:
    s = s.replace("<br/>", "")
    s = emoji_pattern.sub("", s)
    return s.strip()


# MARKDOWN EXTRACTION
def extract_from_markdown(text: str, source_name: str):
    """
    Extract:
    - Section names (## xxx)
    - Questions (**Question**)
    - Answers (until next question or section)
    """
    lines = text.splitlines()

    section = None
    qas = []

    question = None
    answer_buffer = []

    sec_pattern = re.compile(r"^##\s+(.*)")
    q_pattern = re.compile(r"^\*\*(.+?)\*\*")

    for line in lines:
        line = initial_clean(line.strip())

        sec_match = sec_pattern.match(line)
        if sec_match:
            if question:
                qas.append({
                    "section": section,
                    "question": question,
                    "answer": "\n".join(answer_buffer).strip()
                })
            section = sec_match.group(1).strip()
            question = None
            answer_buffer = []
            continue

        q_match = q_pattern.match(line)
        if q_match:
            if question:
                qas.append({
                    "section": section,
                    "question": question,
                    "answer": "\n".join(answer_buffer).strip()
                })
            question = q_match.group(1).strip()
            answer_buffer = []
            continue

        if question:
            answer_buffer.append(line)

    if question:
        qas.append({
            "section": section,
            "question": question,
            "answer": "\n".join(answer_buffer).strip()
        })

    return qas


def process_all_files(md_dir: Path):
    all_records = []
    idx = 1

    md_files = list(md_dir.glob("*.md"))
    print(f"Found {len(md_files)} markdown files.")

    for md_file in md_files:
        raw = md_file.read_text(encoding="utf-8")
        qas = extract_from_markdown(raw, md_file.name)

        for qa in qas:
            subdomain = (
                qa["section"].lower().replace(" ", "_")
                if qa["section"] else "general"
            )

            record = {
                "id": f"theory_{idx:05d}",
                "vector_content": qa["answer"],
                "metadata": {
                    "title": qa["question"],
                    "domain": "theory",
                    "subdomain": subdomain,
                    "taxonomy_skill": [subdomain],
                    "raw_topics": [],
                    "source": md_file.name
                }
            }

            all_records.append(record)
            idx += 1

    print(f"Successfully extracted {len(all_records)} Q&A entries.")
    return all_records


# TEXT CLEANING
def clean_text(text: str) -> str:
    if not text:
        return ""

    text = text.replace("\u200b", "").replace("\xa0", " ")
    text = text.replace("<br/>", "")

    text = re.sub(r"[\U00010000-\U0010ffff]", "", text)
    text = re.sub(r"[\u2600-\u27BF\U0001F000-\U0001FAFF]", "", text)
    text = re.sub(r"[\uFE0F\u200D]", "", text)
    text = text.replace("⭐", "")

    text = re.sub(r"!\[.*?\]\(.*?\)", "", text)
    text = re.sub(r"http\S+|www\S+", "", text)

    text = re.sub(r"[#*`_>{}|]", " ", text)
    text = re.sub(r"\$+", "", text)

    text = re.sub(r"\s+", " ", text)
    return text.strip()


def bad_content(text: str) -> bool:
    if not text:
        return True

    t = text.lower().strip()

    bad_phrases = [
        "answer here", "refer to above", "tbd", "todo",
        "see above", "shown below", "image", "figure"
    ]

    if re.search(r"\bq\d+\b", t):
        return True

    if len(t) < 80:
        return True

    return any(p in t for p in bad_phrases)


def normalize_title(title: str) -> str:
    if not title:
        return ""

    title = clean_text(title)
    title = re.sub(r"^\d+\.\s*", "", title)

    if title.endswith(":"):
        title = f"What is {title[:-1].strip()}?"

    if not title.endswith("?"):
        title = f"{title}?"

    return title.replace("??", "?").strip()


def normalize_subdomain(subdomain: str) -> str:
    if not subdomain:
        return "general"

    subdomain = subdomain.lower().strip()

    bad_map = {
        "sql_questions": "sql",
        "ml_questions": "machine_learning",
        "dbms": "databases"
    }

    return bad_map.get(subdomain, subdomain)


def route_subdomain(text: str) -> str:
    t = text.lower()

    if any(k in t for k in ["bias", "imbalance", "smote", "minority", "majority"]):
        return "fairness_and_imbalance"

    if any(k in t for k in ["cluster", "clustering", "kmeans", "dbscan", "silhouette"]):
        return "clustering"

    if any(k in t for k in ["primary key", "foreign key", "table", "schema", "database"]):
        return "databases"

    if any(k in t for k in ["select ", " join ", " where ", " group by", " order by", " sql"]):
        return "sql"

    return "general"


# MAIN EXECUTION
def main():
    all_records = process_all_files(MD_DIR)

    clean_records = []
    dropped = 0
    repaired = 0

    for obj in tqdm(all_records):
        meta = obj.get("metadata", {})
        obj["metadata"] = meta

        content = clean_text(obj.get("vector_content", ""))
        title = normalize_title(meta.get("title", ""))
        raw_subdomain = meta.get("subdomain", "")
        taxonomy = meta.get("taxonomy_skill", []) or []

        if not title or bad_content(content):
            dropped += 1
            continue

        subdomain = normalize_subdomain(raw_subdomain)

        needs_repair = (
            subdomain == "general"
            or (isinstance(taxonomy, list) and "questions_&_answers_##" in taxonomy)
        )

        if needs_repair:
            new_sub = route_subdomain(content + " " + title)
            subdomain = new_sub
            taxonomy = [new_sub]
            repaired += 1
        elif not taxonomy:
            taxonomy = [subdomain]

        obj["vector_content"] = content
        meta["title"] = title
        meta["subdomain"] = subdomain
        meta["taxonomy_skill"] = taxonomy

        clean_records.append(obj)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for obj in clean_records:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print("\nPRODUCTION CLEANING COMPLETE")
    print("Kept records:", len(clean_records))
    print("Dropped records:", dropped)
    print("Repaired subdomains/taxonomy:", repaired)
    print("Output saved to:", OUTPUT_PATH)

    taxonomy_counts = Counter()
    for obj in clean_records:
        taxonomy_counts.update(obj["metadata"].get("taxonomy_skill", []))

    print("\nALL UNIQUE TAXONOMY SKILLS:\n")
    for t, c in taxonomy_counts.items():
        print(f"{t}  --->  {c}")

    print("\nUNIQUE TAXONOMY SKILLS:", len(taxonomy_counts))


if __name__ == "__main__":
    main()
