# DS Interview Preparation Assistant

An intelligent multi-agent planning system that assists candidates in preparing for data science technical interviews. Given the days left, a job description, and a user profile, it can generate a **skill-aware**, **difficulty-controlled**, and **time-constrained** study plan.

---

## 🌟 Features

- **Multi-Agent Planning Architecture**  
A system composed of specialized agents that collaboratively analyze job requirements, retrieve relevant interview problems, and generate structured preparation plans.

- **Skill-Aware Interview Preparation**  
Identifies required skills from job descriptions and user profile, and allocates preparation effort accordingly.

- **Retrieval with Post-Planning Allocation**  
Retrieves a large candidate pool of interview problems with hybrid RAG, then enforces joint skill and difficulty constraints during later planning.

- **LangChain + Qdrant Agentic RAG**  
Uses OpenAI embeddings, Qdrant vector search, BM25 keyword retrieval, metadata filtering, fallback search, and an optional agentic retrieval controller over a normalized interview-question knowledge base.

- **Structured Study Plans**  
Generates executable interview preparation plans based on user preference (e.g., 7 / 14 / 30-day schedules).

- **Retrieval Evaluation**  
Includes official job-posting-grounded evaluation cases and a script for label-based Recall@K, Precision@K, MRR@K, NDCG@K, category hit rate, skill hit rate, and duplicate rate.

## 🏗️ Architecture

### Agent 1 — Skill Extraction & Weighting

* `scripts/Agent1/input_analyzer.py`: Extracts data-science technical skill signals from the job description and the user's self-description using an LLM.
* `scripts/Agent1/skill_mapper.py`: Maps extracted skill keywords to the project's predefined taxonomy skills via an LLM-assisted matching step.
* `scripts/Agent1/weight_allocator.py`: Assigns a weight to each taxonomy skill based on frequency and requirement strength (e.g., must-have vs. preferred).
* `scripts/Agent1/skill_analyzer_agent.py`: Orchestrates the Agent 1 pipeline and outputs the final skill–weight profile.
* Agent 1 now routes LLM calls through `scripts/langchain_llm.py`, while preserving the original extraction, mapping, and weighting logic.

### Scope Planner — Global Plan Constraints

* `scripts/scope_planner_agent.py`: Determines the overall preparation scope, i.e. total questions, difficulty distribution, and per-skill quotas, given skill weights, time budget, and user constraints.
* The Scope Planner also uses the shared LangChain LLM helper, but its allocation behavior is intentionally unchanged.

### Agent 2 — Retrieval

* `scripts/Agent2/langchain_retrieval.py`: Base retriever that searches the normalized knowledge base using Qdrant dense vector search, BM25 keyword search, metadata filtering, fallback search, and deterministic reranking.
* `scripts/Agent2/agentic_retrieval.py`: Controlled agentic RAG wrapper that uses an LLM to plan multiple retrieval queries, calls the base retriever as a tool, checks coverage, retries when needed, and merges candidates.
* Agent 2 now returns an over-retrieved candidate pool instead of treating retrieval output as the final selected questions.

### Agent 3 — Planning & Scheduling

* `scripts/Agent3/Planning_Agent.py`: Selects final tasks from the Agent 2 candidate pool, then generates a day-by-day study plan under skill quota, difficulty, workload, and spacing constraints.
* The planner preserves retrieval metadata such as retrieval score, adjusted score, requested skill, requested quota, and selection reason.
* Core task selection and scheduling are deterministic; the LLM is used only for controlled swap review and summary polishing.

## 🔁 Current End-To-End Flow

```text
User JD + user profile
        |
        v
Agent 1: extract skills and map them to taxonomy labels
        |
        v
Scope Planner: decide total questions, difficulty distribution, and skill quotas
        |
        v
Agent 2: retrieve candidate questions with controlled agentic RAG over Qdrant + BM25
        |
        v
Agent 3: select final questions using quotas and difficulty targets
        |
        v
Agent 3: schedule selected questions across study days
        |
        v
LLM: optional swap review and summary polishing
        |
        v
Final study plan
```

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
constraint_schedule_tasks()
deterministic summaries
optional LLM swap review and polished summaries
```

Benefits:

- Agent 3 receives `skill_plan`, `difficulty_distribution`, `jd_text`, and `user_desc`.
- Final task selection uses skill quota, difficulty targets, retrieval relevance, direct skill match, and deduplication.
- Scheduling considers daily workload, max questions per day, max hard questions per day, max skills per day, and hard-question spacing.
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

## 📊 Evaluation

The project includes a retrieval-focused evaluation setup.

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
Streamlit multi-agent app
```

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
