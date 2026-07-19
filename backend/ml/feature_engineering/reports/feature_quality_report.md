# Feature Engineering Pipeline Quality Report

## Executive Summary
This report summarizes the feature engineering run for the Fake News Detection pipeline. This phase processes the preprocessed dataset and transforms text and metadata into numerical features.

- **Pipeline Version**: 1.0.3
- **Total Runtime**: 1776.67 seconds
- **Dataset Row Count**: 56573
- **Hand-crafted Dense Features**: 29
- **TF-IDF Sparse Representation Size**: 5000 features

---

## Output Artifacts & Integrity
The following artifacts were successfully generated and validated:

| File Name | Rel Path | SHA-256 Hash |
| --- | --- | --- |
| Feature Dataset CSV | `ml/feature_engineering/processed/feature_dataset_v1.csv` | `7374af83249ea0d13ede645699361b16fb426feb37c6c59c16fd001a95d4ed9f` |
| TF-IDF Vectorizer | `ml/feature_engineering/processed/tfidf_vectorizer.joblib` | `2d3915f00a5eb400bd62493c305c6b9166254fec92e94b1070fadd8bcbb6a417` |
| TF-IDF Sparse Matrix | `ml/feature_engineering/processed/tfidf_matrix.joblib` | `aebea267acb017147712c65974718fdf1b2a0968fd1f78a9ee03c6bddbfbc3b8` |

- **Null Values in Feature Cells**: 0
- **Label Integrity Check**: PASSED

---

## Engineered Feature Groups

### 1. Statistical Features
- word count, character count, sentence count, average word length, average sentence length, vocabulary size.

### 2. Readability Features
- Flesch Reading Ease, Flesch-Kincaid Grade, SMOG index, Gunning Fog, Coleman-Liau index.

### 3. Lexical Features
- lexical diversity, unique words, stopword ratio, long word ratio, short word ratio.

### 4. Symbol Features
- digit count, uppercase count, punctuation count, special character count.

### 5. Linguistic Features (spaCy)
- entity count, noun/verb/adjective count, POS distributions.

### 6. Term Representation (TF-IDF)
- Vocabulary range: 1 to 2 n-grams.
- Max features: 5000

---

## Recommendations
1. **Feature Scale**: Noticeable differences exist in feature scales (e.g., character count ranges into thousands, while stopword ratio is between 0 and 1). Machine learning models (e.g., Logistic Regression, SVM) require scaling. Apply StandardScaler or MinMaxScaler prior to model training.
2. **Dense & Sparse Combination**: For modeling, combine the hand-crafted dense features (from `feature_dataset_v1.csv`) with the TF-IDF sparse matrix (from `tfidf_matrix.joblib`) using scipy's `hstack` to form the final model input matrix.
3. **Feature Selection**: Since we generated 5029 total features, perform feature selection (e.g. variance thresholding or tree-based feature importance) in Phase 5 to prevent overfitting.

*Report automatically generated on 2026-07-18 23:40:06*
