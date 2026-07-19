# Phase 5 – Feature Selection Quality Report

*Generated on: 2026-07-19 00:00:37*

## Executive Summary

This report details the execution and results of the **Feature Selection** phase. The primary objective was to reduce the high-dimensional feature space (engineered dense metrics + TF-IDF textual features) to a highly predictive, low-redundancy subset for the subsequent model training phase.

### Feature Space Compression
| Metric | Input Count | Selected Count | Reduction % |
| :--- | :---: | :---: | :---: |
| **Dense Engineered Features** | 29 | 19 | 34.48% |
| **TF-IDF Text Features** | 5000 | 264 | 94.72% |
| **Total Feature Space** | 5029 | 283 | 94.37% |

- **Total Time Elapsed:** 28.26 seconds
- **Merger Strategy:** `voting` (threshold: 2)

## Verification & Integrity Status

> [!NOTE]
> **VERIFICATION STATUS: PASSED**
> All structural, alignment, and data-integrity checks completed successfully. Sample order and label mapping are verified as preserved row-for-row.

## Selected Features Overview

### Selected Dense Features (19)
`lex_diversity`, `lex_long_word_ratio`, `lex_short_word_ratio`, `lex_stopword_ratio`, `ling_entity_count`, `ling_pos_adj_ratio`, `ling_pos_adv_ratio`, `ling_pos_noun_ratio`, `ling_pos_pron_ratio`, `ling_pos_verb_ratio`, `read_coleman_liau`, `read_flesch_kincaid_grade`, `read_flesch_reading_ease`, `read_gunning_fog`, `read_smog`, `stat_avg_sentence_length`, `stat_avg_word_length`, `stat_char_count`, `sym_special_char_count`

### Sample of Selected TF-IDF Features (264 total)
`tfidf_academic institution`, `tfidf_accord`, `tfidf_across online`, `tfidf_actually`, `tfidf_agreement`, `tfidf_aim`, `tfidf_alert`, `tfidf_allege`, `tfidf_also`, `tfidf_america`, `tfidf_assert`, `tfidf_authorize`, `tfidf_authorize distribution`, `tfidf_back`, `tfidf_black`, `tfidf_britain`, `tfidf_call`, `tfidf_campaign`, `tfidf_china`, `tfidf_circulate`, `tfidf_claim`, `tfidf_clinton`, `tfidf_code`, `tfidf_code ind`, `tfidf_com`, `tfidf_come`, `tfidf_commission`, `tfidf_completely`, `tfidf_constitute`, `tfidf_constitute official`
 ... *and 234 more.*

## Selector Contributions and Performance

| Selector | Enabled | Selected Count | Runtime (s) |
| :--- | :---: | :---: | :---: |
| Variance | True | 71 | 0.0994s |
| Correlation | True | 5013 | 0.1270s |
| Mutual Information | True | 165 | 10.4116s |
| Chi Square | True | 215 | 0.0843s |
| Random Forest | True | 100 | 1.2570s |
| Rfe | True | 30 | 2.5062s |

## Top Selected Feature Rankings

### Variance - Top 10 Features
| Rank | Feature Name | Score/Value |
| :---: | :--- | :---: |
| 1 | `lex_diversity` | 0.049059 |
| 2 | `lex_long_word_ratio` | 0.017738 |
| 3 | `ling_pos_adv_ratio` | 0.015625 |
| 4 | `lex_stopword_ratio` | 0.014491 |
| 5 | `lex_short_word_ratio` | 0.013583 |
| 6 | `ling_pos_pron_ratio` | 0.009357 |
| 7 | `read_flesch_reading_ease` | 0.006810 |
| 8 | `ling_pos_adj_ratio` | 0.006593 |
| 9 | `read_smog` | 0.005879 |
| 10 | `read_gunning_fog` | 0.005005 |

### Correlation - Top 10 Features
| Rank | Feature Name | Score/Value |
| :---: | :--- | :---: |
| 1 | `ling_pos_adv_ratio` | 0.677499 |
| 2 | `ling_pos_noun_ratio` | 0.481511 |
| 3 | `read_gunning_fog` | 0.382946 |
| 4 | `read_smog` | 0.357544 |
| 5 | `ling_pos_pron_ratio` | 0.345353 |
| 6 | `sym_special_char_count` | 0.320685 |
| 7 | `read_flesch_kincaid_grade` | 0.293396 |
| 8 | `read_flesch_reading_ease` | 0.268600 |
| 9 | `lex_stopword_ratio` | 0.252813 |
| 10 | `lex_long_word_ratio` | 0.210864 |

### Mutual Information - Top 10 Features
| Rank | Feature Name | Score/Value |
| :---: | :--- | :---: |
| 1 | `tfidf_num` | 0.597042 |
| 2 | `tfidf_num num` | 0.414228 |
| 3 | `tfidf_say` | 0.406111 |
| 4 | `lex_stopword_ratio` | 0.370171 |
| 5 | `tfidf_via` | 0.364361 |
| 6 | `ling_pos_adv_ratio` | 0.348032 |
| 7 | `lex_long_word_ratio` | 0.336760 |
| 8 | `tfidf_reuters` | 0.311914 |
| 9 | `read_coleman_liau` | 0.305294 |
| 10 | `stat_avg_word_length` | 0.295232 |

### Chi Square - Top 10 Features
| Rank | Feature Name | Score/Value |
| :---: | :--- | :---: |
| 1 | `ling_pos_adv_ratio` | 2522.454447 |
| 2 | `tfidf_via` | 890.577247 |
| 3 | `tfidf_medium platform` | 872.391003 |
| 4 | `tfidf_fully lock` | 854.899226 |
| 5 | `tfidf_please forward` | 854.895934 |
| 6 | `tfidf_misinfo` | 854.879187 |
| 7 | `tfidf_system fully` | 854.871359 |
| 8 | `tfidf_forward widely` | 854.871358 |
| 9 | `tfidf_immediately system` | 854.871358 |
| 10 | `tfidf_ind num` | 854.871358 |

### Random Forest - Top 10 Features
| Rank | Feature Name | Score/Value |
| :---: | :--- | :---: |
| 1 | `tfidf_reuters` | 0.128405 |
| 2 | `tfidf_via` | 0.089266 |
| 3 | `ling_pos_adv_ratio` | 0.059454 |
| 4 | `lex_stopword_ratio` | 0.037444 |
| 5 | `sym_special_char_count` | 0.033533 |
| 6 | `tfidf_image` | 0.033169 |
| 7 | `lex_long_word_ratio` | 0.033026 |
| 8 | `tfidf_feature image` | 0.030574 |
| 9 | `tfidf_say` | 0.024267 |
| 10 | `tfidf_please` | 0.022483 |

### Rfe - Top 10 Features
| Rank | Feature Name | Score/Value |
| :---: | :--- | :---: |
| 1 | `sym_special_char_count` | 1.000000 |
| 2 | `ling_pos_adv_ratio` | 1.000000 |
| 3 | `ling_pos_pron_ratio` | 1.000000 |
| 4 | `tfidf_constitute official` | 1.000000 |
| 5 | `tfidf_contact` | 1.000000 |
| 6 | `tfidf_distribution` | 1.000000 |
| 7 | `tfidf_distribution indexing` | 1.000000 |
| 8 | `tfidf_event` | 1.000000 |
| 9 | `tfidf_fully` | 1.000000 |
| 10 | `tfidf_fully lock` | 1.000000 |

## Selected Features Multicollinearity Analysis

Top correlations among final selected features (absolute value > 0.5):

| Feature 1 | Feature 2 | Pearson Correlation |
| :--- | :--- | :---: |
| `tfidf_authorize distribution` | `tfidf_code ind` | 1.0000 |
| `tfidf_authorize distribution` | `tfidf_constitute official` | 1.0000 |
| `tfidf_authorize distribution` | `tfidf_contemporary event` | 1.0000 |
| `tfidf_authorize distribution` | `tfidf_distribution indexing` | 1.0000 |
| `tfidf_authorize distribution` | `tfidf_event authorize` | 1.0000 |
| `tfidf_authorize distribution` | `tfidf_ind news` | 1.0000 |
| `tfidf_authorize distribution` | `tfidf_index reference` | 1.0000 |
| `tfidf_authorize distribution` | `tfidf_indexing` | 1.0000 |
| `tfidf_authorize distribution` | `tfidf_indexing within` | 1.0000 |
| `tfidf_authorize distribution` | `tfidf_nlp` | 1.0000 |
| `tfidf_authorize distribution` | `tfidf_nlp model` | 1.0000 |
| `tfidf_authorize distribution` | `tfidf_num constitute` | 1.0000 |
| `tfidf_authorize distribution` | `tfidf_official verified` | 1.0000 |
| `tfidf_authorize distribution` | `tfidf_reference code` | 1.0000 |
| `tfidf_authorize distribution` | `tfidf_regard contemporary` | 1.0000 |

## Artifact Signatures & Hash Registry

| File Path | SHA-256 Hash |
| :--- | :--- |
| `selected_feature_dataset_v1.csv` | `9f6591cbe59c477e2c4a19304e07d5665c3fd74c4e34f6fe4a8e75f84be3d14b` |
| `selected_feature_names.json` | `4aa2a26f573c0b6d671a59403917343939cca7fe443dca7aa0eecae266b4f8f6` |
| `feature_rankings.json` | `35757097a1ea1d3470fa461cc36a0d7239fcb35b5d2fb050204d134ceb14f702` |
| `feature_selection_summary.json` | `76aeaca1fb8aaf3f4bc3472dd0f8dd8e658319668124f2df32671c031cd77ae8` |
| `selection_statistics.json` | `1efdc6f80b7e527de41509ed757cd6aa9c7fceefe3b383fe3e96246e70939e9e` |
| `selection_profile.json` | `95ae518673f6590c5eed9dc8f9dd45ba9a0de43bc3a37607e47197f54fc53b9b` |