# NLP Preprocessing Pipeline Quality Report

## Executive Summary
This report summarizes the NLP text preprocessing quality for the master dataset. The pipeline converted raw, noisy article text into normalized, clean tokens suitable for subsequent feature engineering.

- **Pipeline Version**: 1.0.1
- **Runtime**: 1942.15 seconds
- **Dataset Hash**: `06cedb3ae68fe93f6809b08b1d84606b1a39f6b40f3bd2009d0962f85a0a5159`

---

## Dataset Summaries

### Input Dataset Summary
- **Input Path**: `ml/dataset/processed/master_dataset_v1.csv`
- **Total Input Rows**: 56646

### Output Dataset Summary
- **Output Path**: `ml/preprocessing/processed/preprocessed_dataset_v1.csv`
- **Successfully Preprocessed Rows**: 56573
- **Skipped/Failed Rows**: 48
- **Language-Rejected Rows**: 25

---

## Cleaning Effectiveness

### Removed Noise Counts
- **HTML Tags Removed**: 215
- **URLs Removed**: 3670
- **Email Addresses Removed**: 44
- **Emojis Removed**: 10009
- **Special Characters Removed**: 178691
- **Stopwords Removed**: 6916195

### Token Reduction Analysis
- **Average Words Per Input Article**: 289.05
- **Average Tokens Per Output Article**: 172.79
- **Token Reduction Rate**: 40.22%

---

## Language Distribution
The following table shows the detected language distribution of processed articles prior to language filtering:

| Language Code | Article Count |
| --- | --- |
| en | 56621 |
| sq | 3 |
| sw | 4 |
| it | 2 |
| ca | 1 |
| sk | 3 |
| fr | 2 |
| tl | 2 |
| lt | 1 |
| id | 1 |
| pl | 1 |
| af | 1 |
| da | 1 |
| pt | 1 |
| no | 1 |
| es | 1 |

---

## Recommendations
1. **Model Adaptation**: The high token reduction rate (removal of stopwords, punctuation, and short words) is highly suited for sparse bag-of-words or TF-IDF models. For neural models, some stopwords or punctuation might need to be preserved by modifying the configurations in `config/preprocessing_config.yaml`.
2. **Batch Optimization**: If dataset sizes scale past 100k+ rows, consider tuning `pipeline.batch_size` in the config file depending on available RAM.

---
*Report generated automatically on 2026-07-15 23:37:02*