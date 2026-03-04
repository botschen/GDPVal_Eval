# GDPVal-AA Evaluation Framework

GDPVal-AA is Artificial Analysis' evaluation framework for OpenAI's [GDPval dataset](https://huggingface.co/datasets/openai/gdpval).  It assesses language models' capabilities on economically valuable tasks, covering **44 occupations** across key sectors contributing to GDP in the United States.

Paper: [arxiv.org/abs/2510.04374](https://arxiv.org/abs/2510.04374)  
Agent harness: [github.com/ArtificialAnalysis/Stirrup](https://github.com/ArtificialAnalysis/Stirrup)

---

## Overview

The evaluation comprises two stages:

1. **Task Submission** – Models are given a task and required to produce one or more files using an agentic harness with five tools: Web Fetch, Web Search, View Image, Run Shell, and Finish.
2. **Pairwise Grading** – Gemini 3 Pro blindly ranks two submissions for the same task, each created by a different model.

Final ELO scores are computed via a Bradley-Terry model fitted with MLE, bootstrapped confidence intervals (1,000 resamples), and anchored so that GPT-5.1 (Non-Reasoning) = 1,000.

---

## Repository Structure

```
gdpval/
├── elo/
│   ├── bradley_terry.py   # Bradley-Terry MLE ELO computation
│   └── bootstrap.py       # 95 % bootstrapped confidence intervals
├── grading/
│   ├── pairwise_grader.py # Gemini 3 Pro pairwise grader
│   └── sampling.py        # Balanced + active (ELO-informed) match sampling
├── intelligence_index/
│   └── normalize.py       # ELO → Intelligence Index normalization
└── submission/
    └── prompt.py          # Task submission prompt templates & helpers
tests/                     # pytest test suite (45 tests)
```

---

## Installation

```bash
pip install -e ".[dev]"
```

---

## Quick Start

### ELO Calculation

```python
from gdpval.elo.bradley_terry import BradleyTerry
from gdpval.elo.bootstrap import bootstrap_confidence_intervals

# Each tuple is (winner, loser) – ties are excluded per GDPVal-AA methodology.
matches = [
    ("gpt-5.1", "model_b"),
    ("gpt-5.1", "model_c"),
    ("model_b", "model_c"),
]

# Point estimates anchored at GPT-5.1 = 1000
bt = BradleyTerry(anchor_model="gpt-5.1", anchor_elo=1000)
ratings = bt.fit(matches)
print(ratings)  # {'gpt-5.1': 1000.0, 'model_b': ..., 'model_c': ...}

# 95 % bootstrapped confidence intervals (1,000 resamples)
point_estimates, confidence_intervals = bootstrap_confidence_intervals(
    matches, n_bootstrap=1000, random_seed=42
)
```

### Intelligence Index Normalization

```python
from gdpval.intelligence_index.normalize import normalize_elo, normalize_ratings

# Single score: clamp((elo - 500) / 2000, 0, 1)
print(normalize_elo(1400))  # 0.45

# Batch normalization
print(normalize_ratings({"gpt-5.1": 1000, "model_b": 800}))
# {'gpt-5.1': 0.25, 'model_b': 0.15}
```

### Pairwise Match Sampling

```python
from gdpval.grading.sampling import balanced_pairs, active_pairs

models = ["gpt-5.1", "claude-opus-4", "gemini-3-pro", "grok-4"]

# Stage 1: Balanced sampling (each pair once, diverse task coverage)
pairs = balanced_pairs(models, task_ids=["task_01", "task_02"], random_seed=0)

# Stage 2: Active sampling (ELO-informed, prioritises similar-rated models)
current_ratings = {"gpt-5.1": 1000, "claude-opus-4": 1080, "gemini-3-pro": 950, "grok-4": 1120}
pairs = active_pairs(models, current_ratings, n_pairs=50, task_ids=["task_01", "task_02"])
```

### Task Submission Prompt

```python
from gdpval.submission.prompt import build_task_prompt

prompt = build_task_prompt(
    task="Create a bar chart of US GDP by sector.",
    reference_files=["/home/user/gdp_data.csv"],
    finish_tool_name="finish",
)
```

---

## ELO Methodology

- **Model**: Bradley-Terry via MLE (L-BFGS-B) from pairwise win/loss records (ties excluded).
- **Anchor**: GPT-5.1 (Non-Reasoning) fixed at ELO 1,000.
- **Confidence Intervals**: 95 % CI via bootstrap resampling (1,000 iterations).
- **Intelligence Index**: `clamp((ELO - 500) / 2000, 0, 1)`.  ELO scores are frozen at the time of a model's addition.

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Environment Variables

| Variable | Description |
|---|---|
| `GEMINI_API_KEY` | Gemini API key for the pairwise grader |

---

## Sample Evaluation Walkthrough

The following section documents a complete end-to-end evaluation run using the first record fetched from the [openai/gdpval](https://huggingface.co/datasets/openai/gdpval) dataset on HuggingFace.

### Step 1 – Fetch a Sample from the Dataset

The dataset was accessed via the HuggingFace Datasets Server REST API:

```
GET https://datasets-server.huggingface.co/rows
    ?dataset=openai/gdpval&config=default&split=train&offset=0&length=1
```

**Sample metadata**

| Field | Value |
|---|---|
| `task_id` | `83d10b06-26d1-4636-a32c-23f92c57f30b` |
| `sector` | Professional, Scientific, and Technical Services |
| `occupation` | Accountants and Auditors |
| `reference_files` | `reference_files/cc781e4dc0985c8eb327a53ec03b5900/Population v2.xlsx` |
| `deliverable_files` | `deliverable_files/2837faa0a7a6a95f40dfbe45bf66c7fb/Sample v2.xlsx` |

**Task prompt (abridged)**

> You are an auditor and as part of an audit engagement, you are tasked with reviewing and testing the accuracy of reported Anti-Financial Crime Risk Metrics.
>
> The attached spreadsheet titled 'Population' contains Anti-Financial Crime Risk Metrics for Q2 and Q3 2024. Using the data in the 'Population' spreadsheet, complete the following:
> 1. Calculate the required sample size for audit testing based on a **90 % confidence level** and a **10 % tolerable error rate**.
> 2. Perform a **variance analysis** on Q2 and Q3 data (columns H and I).
> 3. **Select a sample** for audit testing based on high-variance metrics, specified entities, risk-weighted metrics, zero-value rows, and geographic/divisional coverage requirements.
> 4. Create a new spreadsheet titled **'Sample'** with the selected rows and the sample-size calculation workings.

---

### Step 2 – Build the Task Submission Prompt

Using `gdpval.submission.prompt.build_task_prompt`:

```python
from gdpval.submission.prompt import build_task_prompt

prompt = build_task_prompt(
    task=sample["prompt"],
    reference_files=sample["reference_files"],
    finish_tool_name="finish",
)
# prompt length: 6,462 characters
```

The prompt wraps the raw task in the full GDPVal-AA agentic harness template, including the available tool list and the `finish` tool instructions.

---

### Step 3 – Pairwise Grading

After collecting model submissions, each pair of submissions is graded by **Gemini 3 Pro** via `PairwiseGrader`.  Submissions are randomly anonymised as "Submission A" and "Submission B" to mitigate position bias.

Simulated grading results for the sample task:

| Match | Grader verdict | Winner | Explanation |
|---|---|---|---|
| gpt-4o vs claude-3.5-sonnet | `A` | gpt-4o | Model A produced a more complete audit sample with all required criteria covered. |
| gpt-4o vs gemini-2.0-flash | `B` | gemini-2.0-flash | Model B provided cleaner variance calculations and correct sample size. |
| claude-3.5-sonnet vs gemini-2.0-flash | `A` | claude-3.5-sonnet | Claude gave better entity coverage. |
| gpt-4o vs gemini-2.0-flash (2nd) | `A` | gpt-4o | gpt-4o dominated with complete FPC formula usage. |

**Collected match records (winner, loser):**

```python
matches = [
    ("gpt-4o",            "claude-3.5-sonnet"),
    ("gemini-2.0-flash",  "gpt-4o"),
    ("claude-3.5-sonnet", "gemini-2.0-flash"),
    ("gpt-4o",            "gemini-2.0-flash"),
]
```

---

### Step 4 – Bradley-Terry ELO Computation

```python
from gdpval.elo.bradley_terry import BradleyTerry

bt = BradleyTerry(anchor_model="gpt-4o", anchor_elo=1000)
ratings = bt.fit(matches)
```

| Model | ELO |
|---|---|
| gpt-4o | **1000.0** (anchor) |
| claude-3.5-sonnet | 927.1 |
| gemini-2.0-flash | 854.2 |

---

### Step 5 – Bootstrapped Confidence Intervals

```python
from gdpval.elo.bootstrap import bootstrap_confidence_intervals

point_est, ci = bootstrap_confidence_intervals(
    matches, n_bootstrap=500, random_seed=42,
    anchor_model="gpt-4o", anchor_elo=1000,
)
```

| Model | ELO | 95 % CI (lower) | 95 % CI (upper) |
|---|---|---|---|
| gpt-4o | 1000.0 | 1000.0 | 1000.0 |
| claude-3.5-sonnet | 927.1 | -3,227.3 | 6,437.4 |
| gemini-2.0-flash | 854.2 | -6,986.2 | 4,344.3 |

> **Note:** Wide confidence intervals are expected with only 4 match records; a production run typically uses hundreds to thousands of comparisons per model.

---

### Step 6 – Intelligence Index

```python
from gdpval.intelligence_index.normalize import normalize_ratings

intel = normalize_ratings(ratings)
# Formula: clamp((ELO - 500) / 2000, 0, 1)
```

| Model | Intelligence Index |
|---|---|
| gpt-4o | 0.2500 |
| claude-3.5-sonnet | 0.2136 |
| gemini-2.0-flash | 0.1771 |

---

### Step 7 – Balanced Match Sampling for Next Round

```python
from gdpval.grading.sampling import balanced_pairs

models = ["gpt-4o", "claude-3.5-sonnet", "gemini-2.0-flash"]
pairs = balanced_pairs(models, task_ids=[sample["task_id"]], random_seed=0)
```

Generated pairings for the next evaluation round:

| Model A | Model B | Task ID |
|---|---|---|
| gemini-2.0-flash | claude-3.5-sonnet | `83d10b06-...` |
| gpt-4o | gemini-2.0-flash | `83d10b06-...` |
| claude-3.5-sonnet | gpt-4o | `83d10b06-...` |

---

### Rubric Summary

The sample task has **38 rubric criteria** worth a maximum of **63 points**, including:

| Points | Criteria |
|---|---|
| 2 | Deliverable is an Excel workbook named `Sample` (.xlsx/.xls/.xlsm) |
| 2 | Workbook contains a `Sample Size Calculation` worksheet |
| 2 | Sample Size Calculation shows 90 % confidence & 10 % tolerable error |
| 2 | Correct population size N used in the sample formula |
| 2 | Attribute sampling formula with z=1.645, p=0.5, e=0.10, and FPC applied |
| 2 | Column I computes QoQ variance as (Q3−Q2)/Q2 |
| 2 | Column J flags sampled rows with `1` |
| 2 | Sample count S ≥ required sample size R |
| 2 | At least one row with \|variance\| ≥ 20 % is included |
| 2 × 5 | Required entity coverage (Italy CB, Greece CB, Luxembourg IB, Brazil CB, UAE PB) |
| 2 × 2 | Required metric coverage (Total Clients, HR Clients) |
| 2 × 2 | Full Division and Sub-Division coverage |
| 5 | Overall formatting and style |
| … | (and 24 further criteria) |
