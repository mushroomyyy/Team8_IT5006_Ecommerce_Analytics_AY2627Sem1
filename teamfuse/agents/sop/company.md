# Company brief

Single source of truth for what this team is, what it builds, and who it
is for. Every agent loads this at start-up.

## What the company does

Team 8 is an NUS IT5006 (AY2026/27 Sem 1) student team doing e-commerce
analytics on the public Olist Brazilian marketplace dataset. The goal is to
understand order behaviour and predict late deliveries so sellers and the
marketplace can act before a delivery misses its promised date.

## The product

Two deliverables live in the parent repo (`../` from `teamfuse/`):

* Phase 1: a Streamlit EDA dashboard (`phase1_eda_dashboard/`, deployed
  copy in `streamlit_release/`) over the bundled compressed Olist data.
* Phase 2: late-delivery modelling in `phase2/` — a classification track
  (`model_dev_v2.ipynb`, `classification_eval.py`,
  `classification_evaluation.ipynb`) and a capped lead-time regression
  track (`regression_dev.ipynb`), sharing helpers in `phase2/src/`
  (`data`, `features`, `labels`, `splits`, `evaluation`).

## Who it is for

The IT5006 teaching staff (grading) and the team itself. Questions come
from teammates asking about data, methods and results. There are no
external customers and no production database.

## Positioning one-liner

Predicting late Olist deliveries from order, seller and product signals,
with time-aware evaluation that avoids leakage.

## How agents should use this

* Ground every answer in files in the parent repo; cite the path.
* Treat the dataset as the only data source. Never invent numbers; run the
  code or read notebook outputs and say which.
* Prefer chronological (time-aware) splits, as `phase2/src/splits.py` does.
* Work on feature branches; never push to `main` directly.
