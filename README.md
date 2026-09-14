# Zepto Data Intelligence Project

## Overview

This project is a complete data intelligence solution for Zepto, covering data pipeline, analytics, and an offline customer support assistant.

The project contains three main modules:

1. Data Pipeline
2. Analytics
3. Support Assistant

---

## Project Structure

```text
zepto-data-intelligence-project/
│
├── data_pipeline/
│   └── scraper.py
│
├── analytics/
│   ├── 01_eda.py
│   ├── titanic.csv
│   ├── classification_metrics.csv
│   ├── final_classification_results.csv
│   ├── imbalance_comparison.csv
│   ├── regression_metrics.csv
│   ├── titanic_best_pipeline.joblib
│   └── charts/
│
├── support_assistant/
│   ├── docs/
│   ├── chroma_db/
│   ├── create_docs.py
│   ├── ingest.py
│   ├── graph.py
│   ├── prompt.py
│   ├── schema.py
│   ├── main.py
│   ├── Dockerfile
│   └── README.md
│
├── query_outputs/
├── books.db
├── .env
├── .gitignore
└── README.md