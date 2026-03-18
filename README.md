# 🧠 Agentic RAG Architectures: Mapping the Efficiency Frontier

> **Master's Thesis Implementation** — Empirical benchmarking of orchestration patterns for multi-hop question answering. This repository demonstrates that intelligent architecture can substitute for model scale: a networked Small Language Model (SLM) outperforms a monolithic Large Language Model (LLM) at 62% lower cost.

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/downloads/)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o--mini-green.svg)](https://openai.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-Cyclic%20Graphs-orange.svg)](https://github.com/langchain-ai/langgraph)
[![Qdrant](https://img.shields.io/badge/Qdrant-Vector%20DB-red.svg)](https://qdrant.tech/)

---

## 📋 Table of Contents

- [Research Overview](#-research-overview)
- [Key Findings](#-key-findings)
- [Architecture Implementations](#-architecture-implementations)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Data Setup](#-data-setup)
- [Quick Start](#-quick-start)
- [Results & Analysis](#-results--analysis)
- [Citation](#-citation)

---

## 🎯 Research Overview

### The Problem

Standard Retrieval-Augmented Generation (RAG) fails on **multi-hop reasoning tasks** — queries that require synthesizing information across multiple documents. While "Agentic" architectures (iterative loops, self-correction, multi-agent debate) close this reasoning gap, they impose **prohibitive latency and cost overheads** that hinder enterprise adoption.

**Example Query:** *"Which writer of 'The Office' also starred in a film directed by Quentin Tarantino?"*

- **Standard RAG:** Searches for documents matching both "The Office" AND "Tarantino" simultaneously → finds neither the connecting writer (B.J. Novak) nor the specific film (*Inglourious Basterds*).
- **Agentic RAG:** Decomposes the task sequentially → searches Office writers → identifies B.J. Novak → searches his filmography → confirms the Tarantino connection.

The trade-off is real: agentic loops solve the reasoning problem, but increase token consumption and latency **linearly** with every additional step.

### Research Questions

**RQ1 — The Trade-off Analysis:** How do agentic architectures compare to naive RAG across the accuracy–latency–cost space? What is the "Price of Intelligence"?

**RQ2 — The Adaptive Solution:** Can a confidence-based routing mechanism achieve near-state-of-the-art accuracy while minimising computational cost by dynamically filtering simple queries to cheaper pipelines?

**RQ3 — The Architectural Hypothesis:** Can orchestrated Small Language Models outperform a monolithic Large Language Model? Does system topology matter more than raw parameter count?

---

## 🏆 Key Findings

### Performance Hierarchy

| Architecture | F1 Score | EM | Avg. Cost | Avg. Latency | Crash Rate | Key Insight |
|---|---|---|---|---|---|---|
| **Network (Debate)** | **58.4%** | **44.8%** | $0.00075 | 9.59 s | 2.0% | Multi-agent debate achieves SOTA |
| **Self-Correcting** | **57.3%** | **43.9%** | $0.00100 | 10.55 s | 6.7% | Strong accuracy, but mechanically unstable |
| **Hybrid (T=0.6)** | **55.0%** | 39.4% | **$0.00052** | 9.89 s | **0.0%** | **Best ROI:** 94% of SOTA at 31% lower cost |
| Baseline (4o) | 54.4% | 40.7% | $0.00196 | 6.81 s | **0.0%** | Large model, but most expensive |
| Supervisor | 52.9% | 40.8% | $0.00084 | 10.10 s | 7.8% | Hierarchical delegation, highest crash rate |
| ReAct | 52.2% | 40.0% | $0.00023 | 6.27 s | 2.1% | Standard iterative loop |
| Baseline (4o-mini) | 48.8% | 36.4% | $0.00012 | 6.28 s | **0.0%** | Cheapest, but limited reasoning |
| Plan-Execute | 39.7% | 24.8% | $0.00048 | 9.13 s | 0.5% | Static planning fails on multi-hop |

### Critical Insights

#### 1. Architecture > Scale *(RQ3 Validated)*

The **Network agent** (`gpt-4o-mini`) outperforms the **Baseline 4o** (`gpt-4o`) by **4.0 F1 percentage points** (58.4% vs. 54.4%) while costing **62% less** ($0.00075 vs. $0.00196). A coordinated system of small experts beats a single large model.

#### 2. The Hybrid Sweet Spot *(RQ2 Validated)*

The **Hybrid Adaptive Agent** achieves 55.0% F1 — capturing 94% of SOTA performance — at only $0.00052 per query through confidence-based routing (threshold δ = 0.6, calibrated on a held-out set):
- **58.7% of queries** are handled by the cheap baseline (instant answers)
- **41.3% of queries** are escalated to the Network agent (complex reasoning)

The Hybrid Agent is the only agentic architecture with a 0.0% crash rate, making it the most mechanically stable option for unsupervised deployment.

#### 3. The Reasoning Barrier *(RQ1 Discovery)*

Agentic loops reduce *retrieval* failures by 41% relative to the Baseline (from 41.6% to 24.7%), confirming that iterative search effectively solves the cold start problem. However, the **conditional reasoning failure rate** (measured only on queries where the correct document was successfully retrieved) remains approximately constant at **31–36%** across all `gpt-4o-mini` architectures, with no trend correlated to architectural complexity. Architecture can fix *search*, but it cannot fix *comprehension*. These agents are context retrieval engines, not reasoning engines. Breaking through this ceiling requires better foundation models, not more complex orchestration graphs.

#### 4. The Router Tax

Adaptive systems pay a **latency penalty** due to sequential execution (classify → potentially escalate). In aggregate throughput scenarios this overhead is more than offset by cost savings.

#### 5. Cyclic Instability

All cyclic architectures exhibit non-trivial crash rates (2.0%–7.8%), where the agent exhausts its step budget without converging on an answer. Only the Baseline architectures and the Hybrid Agent maintain a 0.0% crash rate.

---

## 🏗️ Architecture Implementations

### 1. Baseline RAG — `agent_baseline.py`

**Pattern:** Retrieve → Generate (single-shot)  
**Control flow:** Linear DAG  
**Use case:** High-volume, cost-sensitive workloads

```
User Query → Vector Search (k=5) → LLM Generation → Answer
```

Fastest and cheapest architecture. No multi-hop reasoning capability. Also tested with `gpt-4o` to isolate the effect of model scale vs. architecture.

---

### 2. ReAct Agent — `agent_react.py`

**Pattern:** Thought → Action → Observation (iterative loop)  
**Control flow:** Cyclic state machine  
**Use case:** General-purpose QA with moderate complexity

```
Query → [Reason → Tool Call → Observe]ⁿ → Final Answer
```

Balanced performance (52.2% F1). Adapts retrieval depth dynamically until confident. Crash rate: 2.1%.

---

### 3. Plan-and-Execute — `agent_plan_execute.py`

**Pattern:** Plan → Execute Steps → Replan (static DAG)  
**Control flow:** Directed acyclic graph with optional replanning  
**Use case:** Explainable reasoning for debugging

Lowest F1 (39.7%) due to open-loop planning brittleness. Flawed initial plans propagate unchecked, causing severe context pollution at the synthesis step.

---

### 4. Supervisor Agent — `agent_supervisor.py`

**Pattern:** Manager delegates to specialist workers  
**Control flow:** Hub-and-spoke hierarchy  
**Use case:** Modular, extensible architectures

Separates high-level planning (supervisor) from low-level execution (researchers). High variance (σ = 4.06%) due to "router flicker" — the supervisor occasionally oscillates between routing decisions. Highest crash rate in the benchmark (7.8%).

---

### 5. Self-Correcting Agent — `agent_self_correct.py`

**Pattern:** Generate → Critique → Refine (iterative feedback)  
**Control flow:** Cyclic graph with rejection loops  
**Use case:** High-accuracy tasks that can tolerate longer latency

Strong accuracy (57.3% F1), but significant crash rate (6.7%) from infinite critique loops. A strict search limit and forced approval after three rejections mitigate this.

---

### 6. Network (Debate) Agent — `agent_network.py`

**Pattern:** Proposer ↔ Critic → Judge (multi-agent adversarial debate)  
**Control flow:** Cyclic debate with fixed turns (4 rounds)  
**Use case:** Maximum accuracy; batch/offline processing

```
Query → Proposer → Critic → Proposer → Critic → Judge → Answer
```

Highest F1 (58.4%) with a low crash rate (2.0%). The role separation (Proposer, Critic, Judge) prevents sycophantic agreement and establishes the SOTA ceiling in this study. Directly validates Architecture > Scale.

---

### 7. Hybrid Adaptive Agent — `agent_hybrid.py` ⭐

**Pattern:** Confidence-based router → fast path or slow path  
**Control flow:** Adaptive conditional branching  
**Use case:** Production deployment — optimal cost-performance ratio

```
Query → Router + Confidence Score (δ)
  ├─ δ ≥ 0.6  →  Baseline (Fast Path)   [58.7% of queries]
  └─ δ < 0.6  →  Network Agent (Slow Path) [41.3% of queries]
```

Best overall ROI: 55.0% F1 at $0.00052 with a 0.0% crash rate. The threshold δ = 0.6 was determined empirically via sensitivity analysis on a held-out calibration set of 100 questions (disjoint from the evaluation set), sweeping δ ∈ {0.5, 0.6, 0.7, 0.8, 0.9}.

---

## 📁 Project Structure

```
agentic-efficiency-frontier/
├── src_thesis/
│   ├── __init__.py
│   ├── agent_baseline.py           # Architecture 1: Single-shot RAG
│   ├── agent_react.py              # Architecture 2: ReAct iterative loop
│   ├── agent_plan_execute.py       # Architecture 3: Plan-and-Execute
│   ├── agent_supervisor.py         # Architecture 4: Hierarchical delegation
│   ├── agent_self_correct.py       # Architecture 5: Generate-Critique loop
│   ├── agent_network.py            # Architecture 6: Multi-agent debate
│   ├── agent_hybrid.py             # Architecture 7: Confidence-based router
│   ├── config.py                   # Paths, models, API keys
│   ├── llm_client.py               # Robust OpenAI client (retry + backoff)
│   ├── retrieval_tool.py           # Qdrant vector search wrapper
│   ├── scoring.py                  # F1, EM, Recall metrics
│   ├── utils.py                    # Parallel execution + incremental .jsonl backups
│   ├── ingest.py                   # Wikipedia JSONL → Qdrant vector indexing
│   ├── load_data.py                # HotpotQA evaluation set preparation
│   ├── evaluate_results.py         # Metrics aggregation + 6 visualisation plots
│   ├── analyze_failure_modes.py    # Conditional reasoning + error classification
│   └── run_all.py                  # Orchestration script (full benchmark run)
├── data/
│   ├── hotpot_eval_1000.json       # Main evaluation set (1,000 multi-hop questions)
│   ├── hotpot_eval_300.json        # Smaller set for quick tests
│   └── hotpot_eval_test.json       # Minimal sample for debugging
├── logs/
│   └── latest/                     # Timestamped results, CSVs, and plots
├── ingest_checkpoint.txt           # State tracker for resumable vector indexing
├── requirements.txt
├── .env.example
└── README.md
```

---

## 🚀 Installation

**Prerequisites:** Python 3.11+, Docker Desktop, OpenAI API key

> **Docker Desktop** is required to run the Qdrant vector database. Install it from [docker.com/products/docker-desktop](https://www.docker.com/products/docker-desktop/) and make sure it is running (whale icon visible in your menu bar / system tray) before executing the `docker run` command below.

```bash
# 1. Clone the repository
git clone https://github.com/yannick0711/agentic-efficiency-frontier.git
cd agentic-efficiency-frontier

# 2. Create and activate virtual environment
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 3. Configure environment variables
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY

# 4. Start Qdrant vector database (requires Docker Desktop to be running)
docker run -d --name qdrant_thesis -p 6333:6333 qdrant/qdrant:v1.9.0
```

---

## 📦 Data Setup

### Evaluation Datasets (included)

The `data/` directory contains three pre-built evaluation sets, all generated from the HotpotQA validation split with a fixed random seed (`seed=42`) for reproducibility:

| File | Size | Purpose |
|---|---|---|
| `hotpot_eval_1000.json` | 1,000 questions | Main evaluation set (used in all reported results) |
| `hotpot_eval_300.json` | 300 questions | Quick validation runs |
| `hotpot_eval_test.json` | 100 questions | Threshold calibration (held-out, disjoint from eval set) |

These files are included in the repository. To regenerate them from scratch (produces identical output), run `python -m src_thesis.load_data`.

### Wikipedia Abstracts Corpus (download required)

The vector database must be populated with 5.2 million Wikipedia abstracts before running the agents. This corpus is too large (~2 GB compressed) to include in the repository.

**Step 1 — Download the HotpotQA Wikipedia dump into `data/`:**
```bash
# Using wget:
wget -P data/ https://nlp.stanford.edu/projects/hotpotqa/enwiki-20171001-pages-meta-current-withlinks-abstracts.tar.bz2

# Alternative using curl (if wget is not installed):
curl -L -o data/enwiki-20171001-pages-meta-current-withlinks-abstracts.tar.bz2 https://nlp.stanford.edu/projects/hotpotqa/enwiki-20171001-pages-meta-current-withlinks-abstracts.tar.bz2
```

**Step 2 — Extract into `data/`:**
```bash
tar -xjf data/enwiki-20171001-pages-meta-current-withlinks-abstracts.tar.bz2 -C data/
```

This creates `data/enwiki-20171001-pages-meta-current-withlinks-abstracts/` with subfolders (`AA/`, `AB/`, ...) containing compressed `.bz2` files.

**Step 3 — Convert to a single JSONL file:**
```bash
find data/enwiki-20171001-pages-meta-current-withlinks-abstracts -name "*.bz2" | sort | while read f; do bzcat "$f"; done > data/wiki_abstracts.jsonl
```

This decompresses and concatenates all files into `data/wiki_abstracts.jsonl` (~5.2M lines). This can take a few minutes to complete.

To verify the output:
```bash
wc -l data/wiki_abstracts.jsonl
# Expected: 5233329
```

**Step 4 — Index into Qdrant (~3–4 hours, ~$20 one-time embedding cost):**
```bash
python -m src_thesis.ingest
```

The ingestion process supports checkpointing. If interrupted, re-running the command will resume from where it stopped.

> **Note:** The embedding cost is a one-time expense. Once the vector database is populated, all agent experiments can be run without additional embedding costs. You can optionally delete the raw dump files afterwards to free disk space:
> ```bash
> rm -rf data/enwiki-20171001-pages-meta-current-withlinks-abstracts/
> rm data/enwiki-20171001-pages-meta-current-withlinks-abstracts.tar.bz2
> ```

---

## ⚡ Quick Start

> **Cost estimate:** Running the full benchmark (all 7 agents × 1,000 questions) costs approximately **$10–15 USD** in OpenAI API credits. To verify the pipeline works first, run a single agent on the small test set.

**Verify setup with a quick test:**

```bash
# Run one agent on the small test file to confirm everything works
BASELINE_K=5 python -m src_thesis.agent_baseline
```

**Run the full benchmark:**

```bash
# Executes all 7 agents sequentially, then generates the evaluation report
python -m src_thesis.run_all
```

Results are saved to `logs/<run_id>/`. Incremental `.jsonl` backups are written during execution to prevent data loss on interruption.

**Run a single agent:**

```bash
python -m src_thesis.agent_hybrid
python -m src_thesis.agent_network
python -m src_thesis.agent_react
# etc.
```

When running agents individually (not via `run_all.py`), results are saved to `logs/latest/` by default.

**Evaluate results and generate plots:**

After running the agents, use the evaluation scripts to generate metrics, visualizations, and error analysis:

```bash
# Generate comparative metrics table and 6 visualisation plots
python -m src_thesis.evaluate_results

# Run failure mode classification (retrieval vs. reasoning vs. crash)
python -m src_thesis.analyze_failure_modes
```

By default, both scripts read from `logs/latest/`. If you used `run_all.py`, results are in a timestamped folder (e.g., `logs/run_20260214_143052/`). To point the evaluation scripts at a specific run:

```bash
THESIS_RUN_ID=run_20260214_143052 python -m src_thesis.evaluate_results
THESIS_RUN_ID=run_20260214_143052 python -m src_thesis.analyze_failure_modes
```

---

## 📊 Results & Analysis

### The Efficiency Frontier

A system is Pareto-optimal if no alternative achieves higher accuracy at lower cost. The empirical Pareto frontier defines three deployment tiers:

| Budget Tier | Recommended Architecture | Rationale |
|---|---|---|
| Cost-constrained (< $0.0003) | Baseline (4o-mini) | Cheapest viable option |
| Balanced (≈ $0.0005) | Hybrid Agent (δ=0.6) | Best cost-performance ratio, 0.0% crash rate |
| Performance-critical (> $0.0007) | Network Agent | SOTA accuracy (58.4% F1) |

### Conditional Error Analysis

| Architecture | Retrieval Failure | Absolute Reasoning Error | Conditional Reasoning Error | Crash Rate |
|---|---|---|---|---|
| Baseline (4o-mini) | 41.6% | 20.8% | 35.6% | 0.0% |
| Baseline (4o) | 38.8% | 18.6% | 30.4% | 0.0% |
| Plan-Execute | 53.6% | 20.3% | 44.2% | 0.5% |
| ReAct | 37.8% | 18.7% | 31.1% | 2.1% |
| Supervisor | 28.3% | 21.6% | 33.8% | 7.8% |
| Hybrid (T=0.6) | 36.3% | 22.6% | 35.5% | 0.0% |
| Network | 28.6% | 23.2% | 33.4% | 2.0% |
| Self-Correcting | 24.7% | 22.7% | 33.1% | 6.7% |

*Conditional Reasoning Error isolates the model's synthesis capability by measuring error rate only on questions where the correct supporting documents were successfully retrieved.*

### Deployment Recommendations

**Scenario A — Customer Support Chatbot**  
Constraint: Latency < 7 s → **Baseline (4o-mini)**  
Agentic loops (9–11 s) violate user experience tolerances.

**Scenario B — Internal Knowledge Base**  
Constraint: Cost efficiency at scale → **Hybrid Agent (δ=0.6)**  
Dynamic routing eliminates unnecessary compute on routine queries. 0.0% crash rate ensures mechanical stability.

**Scenario C — Compliance & Reporting**  
Constraint: Maximum accuracy, batch processing → **Network Agent**  
SOTA F1 (58.4%); low crash rate (2.0%); latency is irrelevant for offline jobs.

---

## 📖 Citation

If you use this work in your research, please cite:

```bibtex
@mastersthesis{elss2026agentic,
  title     = {Comparative Evaluation of Orchestration Patterns for Agentic Retrieval-Augmented Generation},
  author    = {Elß, Yannick},
  year      = {2026},
  school    = {Humboldt-Universität zu Berlin},
  type      = {Master's Thesis},
  note      = {Code: https://github.com/yannick0711/agentic-efficiency-frontier}
}
```

---

## 🙏 Acknowledgments

- **HotpotQA Dataset:** Yang et al. (EMNLP 2018)
- **LangGraph:** LangChain team, for cyclic graph orchestration primitives

---

## 📄 License

MIT License — see `LICENSE` for details.
