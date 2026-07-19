# Dataset Quality Report

## Executive Summary
This report summarizes the dataset quality analysis for the standardized Master Dataset used in downstream NLP preprocessing and model training.

- **Dataset Version:** v2
- **Pipeline Runtime:** 99.01 seconds
- **Report Generated Time:** 2026-07-15 22:25:52
- **Integrity Status:** Verification Completed

---

## Dataset Summary Metrics
| Metric | Count | Percentage |
| :--- | :--- | :--- |
| **Total Raw Articles** | 64908 | 100.0% |
| **Final Cleaned Articles** | 56646 | 87.27% |
| **Real Articles (Class 0)** | 29256 | 51.65% |
| **Fake Articles (Class 1)** | 27390 | 48.35% |
| **Rows Dropped (Missing Mandatories)** | 0 | 0.0% |
| **Rows Dropped (Invalid/Corrupted)** | 641 | 0.99% |
| **Rows Dropped (Duplicates)** | 7621 | 11.74% |

---

## Data Distributions

### Language Distribution
- **en**: 56646

### Top 5 Categories
- **politicsNews**: 11134
- **worldnews**: 9665
- **News**: 9048
- **politics**: 6367
- **Business**: 1265

### Top 5 Sources
- **Vishvas News**: 1251
- **PIB Fact Check**: 1251
- **Factly**: 1251
- **AFP Fact Check India**: 1251
- **BOOM Live**: 1251

---

## Article Text Length Analytics
- **Average Word Count:** 288.68 words
- **Average Character Count:** 1785.96 characters
- **Longest Article ID:** master_37788 (51793 chars)
- **Shortest Article ID:** master_32640 (5 chars)

---

## Missing Value Details (Post-Cleaning)
| Field | Missing Count | Missing Percentage |
| :--- | :--- | :--- |
| id | 0 | 0.0% |
| title | 0 | 0.0% |
| text | 0 | 0.0% |
| label | 0 | 0.0% |
| source | 38184 | 67.4081% |
| category | 0 | 0.0% |
| author | 40901 | 72.2046% |
| published_date | 0 | 0.0% |
| language | 0 | 0.0% |
| url | 56646 | 100.0% |
| dataset_origin | 0 | 0.0% |


---

## Potential Data Quality Issues
1. **Unbalanced Categories**: Some categories represent a much larger portion of the dataset, which could bias class predictions.
2. **Missing Author Metadata**: Author information is missing in a high percentage of rows.
3. **Missing Source URL**: A significant number of records lack original URLs, limiting trace verification.

## Recommendations
- **Stratified Split**: When splitting the data into training and validation sets, use stratified splits based on both the `label` and `dataset_origin` to ensure representation.
- **Deduplication Thresholding**: A duplicate threshold of 0.95 removes highly redundant articles. Adjust this threshold inside `config/dataset_config.yaml` to tighten or loosen deduplication.
