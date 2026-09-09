import os
import json
from pathlib import Path
from urllib.parse import quote

RAW_DIR = Path("data/sql_raw")
OUT_PATH = Path("data/sql_raw_extracted.json")

GITHUB_BASE = "https://github.com/mrinal1704/SQL-Leetcode-Challenge/blob/master/"


# taxonomy skills: difficulty level
def detect_difficulty_tag(path: Path) -> str:
    lower_parts = [p.lower() for p in path.parts]
    if "easy" in lower_parts:
        return "sql_easy"
    if "medium" in lower_parts:
        return "sql_medium"
    if "hard" in lower_parts:
        return "sql_hard"
    return "sql_level_unknown"


# url & backup url
def build_urls(title: str, file_path: Path):
    slug = (
        title.lower()
        .replace(" ", "-")
        .replace("_", "-")
        .replace("--", "-")
        .strip("-")
    )
    leetcode_url = f"https://leetcode.com/problems/{slug}/"

    # github as backup
    rel = file_path.relative_to(RAW_DIR)
    github_url = GITHUB_BASE + quote(str(rel).replace("\\", "/"))

    return leetcode_url, github_url

def extract_solution_raw(raw: str) -> str:

    lines = raw.splitlines()
    solution_lines = []
    in_solution = False

    for line in lines:
        text = line.strip().lower()

        # detect start of solution block
        if text.startswith("-- solution") or text == "solution" or "solution:" in text:
            in_solution = True
            continue

        if in_solution:
            # keep raw SQL
            solution_lines.append(line)

    return "\n".join(solution_lines).strip()

def clean_description(raw: str) -> str:
    lines = raw.splitlines()

    # Remove first line like "-- Question ..."
    if lines and lines[0].strip().startswith("--"):
        lines = lines[1:]

    cleaned = []
    in_solution = False

    for raw_line in lines:
        line = raw_line.strip()

        # skip solution section
        if line.lower().startswith("solution") or "solution:" in line.lower():
            in_solution = True
            continue
        if in_solution:
            continue

        # remove "--" prefix
        if line.startswith("--"):
            line = line[2:].strip()

        if not line:
            continue

        # remove ascii table lines
        if line.startswith("|") or line.startswith("+"):
            continue

        # remove markdown table header
        if "|" in line and ("--" in line or "---" in line):
            continue

        # remove schema lines
        blacklist = [
            "create table", "primary key",
            "column name", "type", "table "
        ]
        if any(k in line.lower() for k in blacklist):
            continue

        cleaned.append(line)

    return "\n".join(cleaned).strip()


def main():
    id_counter = 1
    all_records = []

    with OUT_PATH.open("w", encoding="utf-8") as fout:
        for root, dirs, files in os.walk(RAW_DIR):
            for fname in files:

                # Skip non-sql files
                if not fname.endswith(".sql"):
                    continue

                fpath = Path(root) / fname
                raw_text = fpath.read_text(encoding="utf-8", errors="ignore")

                problem_id = f"SQL_{id_counter}"
                id_counter += 1

                title = (
                    fpath.stem.replace("_", " ").replace("-", " ").title().strip()
                )

                category = "SQL"
                difficulty_tag = detect_difficulty_tag(fpath)
                url, backup_url = build_urls(title, fpath)

                metadata = {
                    "id": problem_id,
                    "title": title,
                    "category": category,
                    "taxonomy_skills": [difficulty_tag],
                    "solution_summary": "",
                    "url": url,
                    "backup_url": backup_url
                }

                description_clean = clean_description(raw_text)
                solution_raw = extract_solution_raw(raw_text)

                record = {
                    "vector_content": description_clean,
                    "solution_raw": solution_raw,
                    "metadata": metadata
                }
                all_records.append(record)

                with OUT_PATH.open("w", encoding="utf-8") as fout:
                    json.dump(all_records, fout, ensure_ascii=False, indent=2)

    print(f"SQL dataset has been extracted → {OUT_PATH}")


if __name__ == "__main__":
    main()
