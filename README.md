# DS Interview Preparation Assistant

A continuously tracking interview learning assistant for data science roles. Given a job description, a user profile and the days left, it builds a **skill-aware**, **difficulty-controlled**, **time-constrained** study plan, then keeps it up to date as the user reports progress: finished and wrong questions, weak topics, and changes to the interview date.

---

## 🌟 Features

- **Agentic workflow: a controller agent over specialized modules**  
The first plan is built by a fixed pipeline (skill analysis → scope planning → retrieval → scheduling). After that, a ReAct controller agent reads the user's message and decides, through tool calling, which module to use: skill analysis, question retrieval, plan generation or plan update.

- **Short-term and long-term memory**  
The current plan, question statuses and conversation are persisted per session with a LangGraph checkpointer. Long-term memory records completed questions, mistakes, self-reported weak skills and preferences across sessions.

- **Feedback loop**  
Feedback updates skill weights with a deterministic rule, filters already-done questions out of retrieval, and re-plans only the remaining days (today and finished questions are never moved).

- **LangChain + Qdrant hybrid / agentic RAG**  
OpenAI embeddings, Qdrant vector search, BM25 keyword retrieval, metadata filtering, and a controlled agentic retrieval loop over a normalized question bank of about 2.9K questions.

- **Evaluation**  
Retrieval metrics (Precision@K, Recall@K, NDCG@K, duplicate rate), a tool-calling evaluation for the controller (success rate on dev and held-out cases), a plan-quality evaluation, and deterministic tests for re-planning, the controller graph and the Streamlit demo.

## 🏗️ Architecture

```text
              START
                ↓
        plan already exists?
       no ↙            ↘ yes
Agent1 → Scope →        controller (LLM + tools) ⇄ tools      ReAct loop, ≤ 5 tool calls per turn
Agent2 → Agent3          ↓ no more tool calls
       ↓                END
      END
(every step is saved by the checkpointer: short-term memory)
```

### Controller agent

* `scripts/controller/graph.py`: the LangGraph graph. First turn → fixed pipeline; later turns → a ReAct controller (`bind_tools`, native function calling) looping with a `ToolNode`.
* Guardrails in code: at most 5 tool calls per turn, `analyze_skills` / `generate_plan` at most once per turn, no parallel tool calls, tool errors returned to the LLM.
* Context management: the LLM sees a compact view (today's and the next two days' numbered questions, progress, weak skills, preferences) instead of the full plan; tools return short summaries while full results stay in the state; earlier turns are trimmed.
* `prompts/controller_system.txt`: decision table for when to use which tool and when not to call a tool.

### Tools (`scripts/controller/tools.py`)

| Tool | Wraps | Used when |
|---|---|---|
| `analyze_skills` | Agent 1 | the target role or JD changes |
| `retrieve_questions` | Agent 2 | more candidate questions are needed |
| `generate_plan` | Scope Planner + Agent 2 + Agent 3 | full re-plan |
| `update_plan` | `scripts/controller/replan.py` (rules, no LLM) | results, weak skills, days left, daily load |

Date arithmetic (`interview_day` → study days) and id resolution are done in code, not by the LLM.

### Memory

* Short-term: LangGraph `SqliteSaver` (`.state/checkpoints.db`), one thread per preparation session.
* Long-term: `scripts/controller/memory.py` (`.state/long_term.db`): `feedback_log` (done / wrong / struggling per question and skill) and `preferences`.
* `scripts/controller/feedback.py`: weights are recomputed from Agent 1's base weights × a factor per skill (×1.5 weak, ×0.8 mastered, from the last 5 records), then renormalized.

### Pipeline modules

* **Agent 1** (`scripts/Agent1/`): extracts skills from the JD and the user profile with an LLM, maps them to the taxonomy, and assigns weights; weights are adjusted by the user's history.
* **Scope Planner** (`scripts/scope_planner_agent.py`): total questions, difficulty mix and per-skill quotas; quotas are capped by how many questions the bank has for each skill and by a 35% max share.
* **Agent 2** (`scripts/Agent2/`): hybrid Qdrant + BM25 retrieval with an optional agentic RAG loop (query planning, coverage check, retry); only candidates labelled with the requested skill and not yet done by the user are kept.
* **Agent 3** (`scripts/Agent3/Planning_Agent.py`): greedy selection by quota and difficulty targets, then `schedule()`: balanced daily counts with a lighter review day, at most one hard question per day, easy → medium ramp. The LLM only writes the daily summaries. `update_plan` reuses the same `schedule()`.

## 🔄 Upgrade Summary

### Data And Knowledge Base

Before:

```text
data/merged1.jsonl
runtime sentence-transformer embeddings
in-memory semantic retrieval
```

After:

```text
data/questions_normalized.jsonl
OpenAI embeddings
local Qdrant vector database
LangChain Document abstraction
```

The normalized dataset uses a flat schema:

```json
{
  "id": "string",
  "type": "coding | theory",
  "category": "SQL | Pandas | Algorithms | ML | Statistics | Product",
  "title": "string",
  "question": "string",
  "answer": "string",
  "difficulty": "easy | medium | hard",
  "taxonomy_skills": ["string"],
  "source": "string",
  "url": "string"
}
```

Benefits:

- Removes inconsistent nested metadata.
- Separates difficulty labels from taxonomy skills.
- Makes coding and theory questions searchable with one schema.
- Makes metadata filtering and evaluation easier.

### Retrieval Logic

Before:

```text
skill name
→ embedding search
→ MMR
→ return exactly the requested number of questions
```

After:

```text
skill + JD + user background
→ LLM retrieval query planning
→ call base Qdrant + BM25 retriever as a tool
→ deterministic coverage check
→ adaptive retry if coverage is weak
→ merge, dedupe, and rerank candidates
→ return a larger candidate pool
```

Benefits:

- Vector search captures semantic similarity.
- BM25 helps exact terms such as SQL window function, A/B testing, causal inference, or random forest.
- Metadata filters can constrain by type, category, difficulty, or taxonomy skill.
- Fallback and adaptive retry avoid empty or overly narrow candidate pools when taxonomy coverage is sparse.
- Retrieval and planning now have cleaner responsibilities: retrieval finds options, planner chooses the final set.

### Planner Logic

Before:

```text
flatten retrieved questions
schedule every retrieved question
optional LLM review and summaries
```

After:

```text
preserve retrieval metadata
build plan constraints
select_final_tasks()
schedule()            # replaced constraint_schedule_tasks() and the LLM swap review
deterministic summaries
optional LLM-polished summaries
```

Benefits:

- Agent 3 receives `skill_plan`, `difficulty_distribution`, `jd_text`, and `user_desc`.
- Final task selection uses skill quota, difficulty targets, retrieval relevance, direct skill match, and deduplication.
- Scheduling balances the number of questions per day (lighter last day for review), allows at most one hard question per day, and ramps difficulty from easy to medium.
- Deterministic summaries are always generated before optional LLM polishing.

### LangChain Unification

Before:

```text
Agent 1 / Scope Planner / Agent 3 used direct OpenAI SDK calls.
Agent 2 used a separate local embedding retriever.
```

After:

```text
Agent 1, Scope Planner, and Agent 3 LLM calls use scripts/langchain_llm.py.
Agent 2 retrieval uses LangChain + OpenAI embeddings + Qdrant.
Agent 2 can optionally use a controlled agentic RAG wrapper around the base retriever.
```

What intentionally did not change:

- Agent 1 prompts.
- Skill mapping contract.
- WeightAllocator logic.
- Scope Planner allocation logic.

### Retrieval Mode Toggle

The main application uses agentic retrieval by default. To switch back to the base hybrid retriever:

```bash
export USE_AGENTIC_RETRIEVAL=false
```

To use agentic retrieval:

```bash
export USE_AGENTIC_RETRIEVAL=true
```

## 🔧 Reproducible workflow

### **1. Data Pipeline**

#### **a. SQL Leetcode Database**

* **Data Source**
  * The raw SQL LeetCode dataset is exported from the public repository: `https://github.com/mrinal1704/SQL-Leetcode-Challenge/blob/master/` 
  * The exported raw files are stored under: `data/sql_raw`

* **Schema Standardization**
  * `data_prep/extract_sql_raw.py` converts each problem into the project standardized JSON schema and writes the result to `data/sql_raw_extracted.json`
  * Each record follows the unified schema:
    
    ```json
    {
        "vector_content": "<cleaned content>",
        "metadata": {
        "id":"",
        "title": "<question title>",
        "category": "SQL",
        "taxonomy_skill": [ ],
        "solution_summary": "<summarized answer>",
        "url": "<leetcode link>",
        "backup_url":"<github link>"
        }
    }
    ```

* **Taxonomy Annotation**
  * `data_prep/annotate.py` uses an LLM to annotate/tag each SQL problem with the pre-defined taxonomy skills
  * The taxonomy definition is stored in: `data/taxonomy_skills.json`
  * The annotated SQL dataset is written to: `data/sql.json`


#### **b. Algorithms/Pandas Leetcode Database**
* **Data Source**
  * The raw Algorithms and Pandas LeetCode dataset is exported from Kaggle: `https://www.kaggle.com/datasets/alishohadaee/leetcode-problems-dataset`
  * The exported raw file is stored under: `data/leetcode_problems_raw.json`, and `data/algorithms_raw.json` is extracted from this file
* **Schema Standardization**
  * `data_prep/algo_preprocessing.py` and `data_prep/pandas_preprocessing.py` converts each problem into the project standardized JSON schema and annotate the taxonomy skills, finally writes the results to `data/algorithms.json` and `data/pandas.json` respectively
  * Each record follows the unified schema based on category:
      ```json
    {
        "vector_content": "<cleaned content>",
        "metadata": {
        "id":"",
        "title": "<question title>",
        "category": "Algorithms",
        "taxonomy_skill": [ ],
        "solution_summary": "<summarized answer>",
        "url": "<leetcode link>",
        "backup_url":""
        }
    }
    ```
    ```json
    {
        "vector_content": "<cleaned content>",
        "metadata": {
        "id":"",
        "title": "<question title>",
        "category": "Pandas",
        "taxonomy_skill": [ ],
        "solution_summary": "<summarized answer>",
        "url": "<leetcode link>",
        "backup_url":""
        }
    }
    ```
    

#### **c. Leetcode Dataset Merge**

  * `data_prep/merge_leetcode.py` merges the SQL dataset with the Algorithms/Pandas LeetCode dataset, aligning both to a unified schema. The merged output is written to: `data/leetcode.json`

#### **d. Theory Database**

* **Data Source**

  * Theory Q&A content is stored as markdown files under: `data/Theory_raw/` (all `*.md` files).

* **Extraction & Cleaning**

  * `data_prep/theory_extraction_and_cleaning.py` parses markdown sections and Q&A blocks, performs text cleaning/normalization, and repairs taxonomy/subdomains when needed.
  * Extraction rules:

    * Section header (`## ...`) is treated as a **subdomain** candidate.
    * Q&A pairs are extracted from `**Question**`-style headings and subsequent lines until the next question/section.

* **Output Format**

  * The processed dataset is written as JSONL to: `data/Theory.jsonl`.
  * Each record follows the unified schema:

    ```json
    {
      "id": "",
      "vector_content": "<cleaned answer text>",
      "metadata": {
        "title": "<normalized question>",
        "domain": "theory",
        "subdomain": "<normalized/routed subdomain>",
        "taxonomy_skill": ["<subdomain>"],
        "raw_topics": [],
        "source": "<markdown filename>"
      }
    }
    ```

* **Taxonomy/Subdomain Repair**

  * If the extracted subdomain is missing/too generic, the script assigns a more specific subdomain using keyword-based routing (e.g., `sql`, `databases`, `clustering`, `fairness_and_imbalance`).

#### **e. Final Normalized Dataset**

The final normalizer combines SQL, Algorithms, Pandas, and theory data into one canonical JSONL file:

```bash
python data_prep/normalize_dataset.py
```

Output:

```text
data/questions_normalized.jsonl
```

This file is the source of truth for the LangChain/Qdrant knowledge base and for evaluation label validation.

### **2. Build The LangChain/Qdrant Knowledge Base**

After `data/questions_normalized.jsonl` is ready, build the local Qdrant vector database:

```bash
python knowledge_base/build_langchain_kb.py
```

This creates:

```text
knowledge_base/vectorstores/questions_qdrant
```

The build script:

- Loads normalized question records.
- Converts each record into a LangChain `Document`.
- Embeds document text with OpenAI embeddings.
- Stores vectors and metadata in a local Qdrant collection.


### **3. Local Installation & Launch**

- Clone and enter the project
```bash
git clone <YOUR_GITHUB_REPO_URL>
cd <YOUR_PROJECT_FOLDER>
```

- To configure environment variables, create a local `.env` file:
```bash
touch .env
```

- Open `.env` and fill in your key:
```bash
OPENAI_API_KEY="YOUR_OPENAI_KEY_HERE"
```

- Create the conda environment for package installation (first-time setup)
```bash
conda env create -f environment.yml
conda activate ds-interviewer
```

- Install Python package requirements if needed
```bash
pip install -r requirements.txt
```

- Run Streamlit by `python -m` to ensure Streamlit runs under the correct conda environment:
```bash
python -m streamlit run demo.py
```

- If Qdrant Cloud is not reachable, use the local vector store in `knowledge_base/vectorstores/`:
```bash
QDRANT_URL= python -m streamlit run demo.py
```

- Demo flow: create a plan (pipeline progress is shown step by step) → mark today's questions done / wrong → tell the assistant what happened ("我实在不会 SQL 窗口函数", "面试提前到后天了") and watch the tool calls and the updated plan → "模拟：进入下一天" moves the plan one day forward for demo purposes.

- Terminal chat with the controller (shows every tool call):
```bash
python -m scripts.controller.graph <thread_id>
```

- Tests (no API key needed):
```bash
python tests/test_replan.py      # re-planning rules
python tests/test_controller.py  # graph wiring and guardrails, scripted LLM
python tests/test_demo.py        # Streamlit demo end to end (AppTest), scripted LLM
```

## 📊 Evaluation

Three evaluations: retrieval quality (below), controller tool calling, and plan quality.

### Tool-Calling Evaluation (controller)

```bash
python eval/evaluate_tool_calling.py --repeats 3
python eval/evaluate_tool_calling.py --cases eval/tool_calling_holdout.json --repeats 3
```

* 22 dev cases and 8 held-out cases, all starting from the same fixture plan; the controller uses the real LLM, retrieval and pipeline modules are faked so only the controller's decisions are scored.
* A case passes when the tool sequence is exactly right (no tool for explanations, vague requests and chit-chat), key arguments are right, and `update_plan` sets no plan-shape argument the user did not ask for.
* Results (gpt-4o-mini, 3 repeats): dev 100% (66/66); held-out 79.2% on its first run, 83.3% at the end (it was used for diagnosis, so this is optimistic). The full iteration history is in `optimize.md`.

### Plan Quality Evaluation

```bash
QDRANT_URL= python eval/evaluate_plan_quality.py --label after
```

Runs the real pipeline on 6 JDs and reports skill match rate, max skill share, quota over supply, daily count range, difficulty gap and latency. Before → after the planner simplification: skill match 0.856 → 1.000, quota over supply 12.2% → 0%, daily range 4.67 → 3.50, 51.9 s → 48.4 s per plan.

### Retrieval Evaluation

The retrieval evaluation setup:

### Evaluation Data

Unified evaluation cases:

```text
eval/eval_cases.json
```

The eval set contains 30 official-company-source-grounded cases from companies such as Apple, Amazon, Google, Microsoft, Netflix, Uber, Airbnb, and Meta.

Each case includes:

```text
query
jd
user_desc
expected_categories
expected_skills
expected_concepts
source_url
```

The `jd` field is a short paraphrase grounded in an official company job posting or official company career guidance page. The expected labels are taxonomy-aligned and validated against `data/questions_normalized.jsonl`.

### Evaluation Modes

The script supports two evaluation modes:

```text
retrieval_only
agent1_retrieval
```

`retrieval_only` evaluates Agent 2 directly:

```text
manual query from eval_cases.json
        |
        v
Agent 2 retrieval
        |
        v
compare top-k results against expected categories and skills
```

`agent1_retrieval` evaluates Agent 1 plus Agent 2:

```text
jd + user_desc
        |
        v
Agent 1 skill extraction and taxonomy mapping
        |
        v
Agent 2 retrieval using Agent 1 skills
        |
        v
compare top-k results against expected categories and skills
```

### Metrics

The evaluation script computes:

```text
Precision@K
Recall@K
MRR@K
NDCG@K
Category Hit Rate
Skill Hit Rate
Duplicate Rate
```

Relevance is label-based:

```text
A retrieved question is relevant if:
retrieved category overlaps expected_categories
OR
retrieved taxonomy_skills overlaps expected_skills
```

This is a weakly supervised retrieval evaluation, not a manually judged question-ID relevance dataset.

The script can evaluate either retriever:

```text
base
agentic
```

`base` means fixed hybrid RAG using Qdrant + BM25. `agentic` means controlled agentic RAG with query planning, multi-query retrieval, coverage checking, and adaptive retry.

### Validate Eval Labels

This checks that the expected labels in `eval/eval_cases.json` exist in the normalized KB:

```bash
python -c "from eval.evaluate_retrieval import load_cases, load_known_labels, validate_case_labels, DEFAULT_CASES_PATH; cases=load_cases(DEFAULT_CASES_PATH); cats, skills=load_known_labels(); validate_case_labels(cases, cats, skills); print('labels valid:', len(cases), 'cases')"
```

### Run Retrieval-Only Evaluation

```bash
python eval/evaluate_retrieval.py \
  --mode retrieval_only \
  --retriever base \
  --topk 10 \
  --output eval/retrieval_only_results.json
```

To evaluate agentic retrieval directly:

```bash
python eval/evaluate_retrieval.py \
  --mode retrieval_only \
  --retriever agentic \
  --topk 10 \
  --output eval/retrieval_only_agentic_results.json
```

### Run Agent1 + Retrieval Evaluation

```bash
python eval/evaluate_retrieval.py \
  --mode agent1_retrieval \
  --retriever base \
  --topk 10 \
  --output eval/agent1_retrieval_results.json
```

To evaluate Agent1 plus agentic retrieval:

```bash
python eval/evaluate_retrieval.py \
  --mode agent1_retrieval \
  --retriever agentic \
  --topk 10 \
  --output eval/agent1_retrieval_agentic_results.json
```

`@10` means each metric is computed over the top 10 retrieved results per case, then averaged across the 30 cases.

## ☁️ GCP Cloud Run Deployment

The recommended cloud deployment is:

```text
GCP Cloud Run
        |
        |-- OpenAI API
        |
        |-- Qdrant Cloud
        |
        v
Streamlit app
```

Note: the current version keeps short-term and long-term memory in local SQLite files under `.state/`. Cloud Run containers are stateless, so before deploying, both stores should move to a managed database (for example Postgres with LangGraph's `PostgresSaver`); otherwise plans and memory are lost when a container restarts. The demo also uses a single local user (`default_user`).

Cloud Run should not depend on the local Qdrant folder. Instead, upload the vector collection to Qdrant Cloud once, then let the deployed app connect to Qdrant Cloud with environment variables.

### Deployment Files

```text
Dockerfile
requirements-cloud.txt
.dockerignore
```

`requirements-cloud.txt` is intentionally smaller than the full local `requirements.txt`, because the deployed app only needs the runtime dependencies for Streamlit, LangGraph, LangChain, OpenAI, Qdrant, and BM25.

### Required Cloud Environment Variables

```text
OPENAI_API_KEY
QDRANT_URL
QDRANT_API_KEY
QDRANT_COLLECTION_NAME
```

Recommended collection name:

```text
ds_interview_questions
```

### 1. Upload The Knowledge Base To Qdrant Cloud

Create a Qdrant Cloud cluster, then run the KB build script with cloud variables:

```bash
export OPENAI_API_KEY="YOUR_OPENAI_KEY"
export QDRANT_URL="YOUR_QDRANT_CLOUD_URL"
export QDRANT_API_KEY="YOUR_QDRANT_API_KEY"
export QDRANT_COLLECTION_NAME="ds_interview_questions"

python knowledge_base/build_langchain_kb.py
```

With `QDRANT_URL` set, the script writes to Qdrant Cloud. Without `QDRANT_URL`, it writes to the local Qdrant path.

### 2. Configure GCP

```bash
gcloud auth login
gcloud config set project YOUR_GCP_PROJECT_ID
gcloud services enable run.googleapis.com cloudbuild.googleapis.com secretmanager.googleapis.com
```

### 3. Store Secrets

```bash
printf "%s" "YOUR_OPENAI_KEY" | gcloud secrets create openai-api-key --data-file=-
printf "%s" "YOUR_QDRANT_API_KEY" | gcloud secrets create qdrant-api-key --data-file=-
```

If the secret already exists, add a new version instead:

```bash
printf "%s" "YOUR_OPENAI_KEY" | gcloud secrets versions add openai-api-key --data-file=-
printf "%s" "YOUR_QDRANT_API_KEY" | gcloud secrets versions add qdrant-api-key --data-file=-
```

### 4. Deploy To Cloud Run

From the project root:

```bash
gcloud run deploy ds-interview-copilot \
  --source . \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars QDRANT_URL="YOUR_QDRANT_CLOUD_URL",QDRANT_COLLECTION_NAME="ds_interview_questions" \
  --set-secrets OPENAI_API_KEY=openai-api-key:latest,QDRANT_API_KEY=qdrant-api-key:latest
```

Cloud Run will build the Docker image from `Dockerfile` and run Streamlit on the `$PORT` assigned by Cloud Run.

### 5. Test The Deployed App

After deployment, GCP prints a service URL:

```text
https://ds-interview-copilot-xxxxx.run.app
```

Open it in a browser, paste a job description and user profile, then generate a study plan.

## 💡 Usage Example
demo url: https://www.loom.com/share/192589aa565d48ef9b2eb106731a264e

<img width="688" height="504" alt="17ae312c8600869a07a8f1f5e201a6ec" src="https://github.com/user-attachments/assets/eb3a1cec-32d1-4e86-bedd-692e1dff524f" />
