
# Data Quality Report

## 1. Dataset Overview

The training dataset (`train.csv`) contains records related to bakery loaf sales. The dataset was inspected to identify data quality issues before beginning data preprocessing and model development.

- **Original number of rows:** 1,036
- **Number of columns:** 12
- **Rows after exact deduplication:** 1,000
- **Target variable:** `loaves_sold`
- **Date range:** 1 January 2026 – 30 June 2026

The dataset contains sales-related information, environmental factors, outlet information, and historical sales-related features.

---

## 2. Missing Values

An inspection of the original dataset revealed no missing values in any of the columns.

No missing-value imputation is currently required.

---

## 3. Duplicate Records

A total of **36 exact duplicate rows** were identified.

- Duplicate groups: 36
- Each duplicate group contained two identical records.
- Rows involved in duplicate groups: 72
- Rows remaining after exact deduplication: 1,000

Exact duplicates should be removed during preprocessing to avoid giving duplicate observations additional weight during model training.

---

## 4. Record ID Integrity

The original dataset contained duplicate records associated with repeated IDs.

After removing exact duplicate rows:

- Total records: 1,000
- Unique `record_id` values: 1,000
- Remaining duplicate IDs: 0

The `record_id` column should be retained for identification and validation purposes but should not automatically be used as a predictive feature.

---

## 5. Date Integrity

The dataset covers the period from **1 January 2026 to 30 June 2026**.

- Unique dates: 181
- Missing calendar dates: 0
- Date range is continuous.

The `date` column should be converted into a proper datetime format during preprocessing. Additional time-based features may be derived later if they are relevant to the prediction task.

---

## 6. Numerical Feature Analysis

The following summary statistics were observed in the dataset:

| Feature | Mean | Standard Deviation | Minimum | Maximum |
|---|---:|---:|---:|---:|
| `price_rs` | 46.687 | 12.679 | 25 | 70 |
| `temp_c` | 25.405 | 11.471 | 5.1 | 44.9 |
| `footfall_index` | 54.600 | 26.490 | 10 | 100 |
| `competitor_shops` | 3.921 | 2.326 | 0 | 8 |
| `shelf_hours` | 9.958 | 3.476 | 4 | 16 |
| `orders_previous_day` | 250.906 | 142.708 | 10 | 499 |
| `loaves_in_closing_register` | 319.864 | 97.691 | 74 | 629 |

The numerical values appear broadly plausible for their respective features. However, their distributions, relationships with the target, and possible outliers should be examined further during exploratory data analysis.

---

## 7. Target Variable Investigation

The target variable is `loaves_sold`.

### Target statistics before cleaning

| Statistic | Value |
|---|---:|
| Mean | 356.301 |
| Standard deviation | 340.534 |
| Minimum | 75 |
| 25th percentile | 251 |
| Median | 324 |
| 75th percentile | 392.25 |
| Maximum | 4,920 |

The maximum value of 4,920 is considerably larger than the typical target values and requires investigation.

### Suspected 10x scaling anomalies

Fifteen records contain target values greater than or equal to 1,000. These values appear potentially affected by a 10x scaling or recording error.

| Record ID | `loaves_sold` | Closing Register |
|---|---:|---:|
| R00024 | 1,870 | 186 |
| R00039 | 2,060 | 214 |
| R00063 | 1,920 | 197 |
| R00142 | 3,940 | 399 |
| R00174 | 4,920 | 497 |
| R00197 | 2,030 | 204 |
| R00416 | 1,920 | 192 |
| R00426 | 1,370 | 128 |
| R00532 | 2,240 | 217 |
| R00555 | 1,390 | 136 |
| R00562 | 3,370 | 338 |
| R00607 | 2,630 | 265 |
| R00770 | 3,920 | 390 |
| R00874 | 3,750 | 367 |
| R01000 | 4,650 | 464 |

Dividing these target values by 10 results in values close to the corresponding closing register values. This strongly suggests a possible scaling error, although the original records have not yet been independently verified.

### Recommended handling

- Do not delete these records automatically.
- Preserve the original target values.
- Flag the 15 records for further investigation.
- During later experimentation, compare model performance using the original targets against a documented correction where the suspected 10x values are divided by 10.
- Do not apply corrections until the hypothesis has been evaluated and documented.

---

## 8. Closing Register Feature Investigation

The feature `loaves_in_closing_register` has a weak positive correlation with the target `loaves_sold`, approximately **0.293**.

Although it is useful for investigating suspicious target values, it should not be used as a model input because it is absent from `holdout_features.csv`.

Using it during training would create a train-test feature mismatch and prevent its use when making predictions on the holdout dataset.

It should therefore be excluded from the final prediction features.

---

## 9. Categorical Data Quality — `day_type`

The `day_type` column contains inconsistent representations of the same categories.

Issues identified include:

- Uppercase and lowercase variations.
- Leading and trailing whitespace.
- Abbreviations such as `fes.` and `wee.`
- Inconsistent formatting of `weekday`, `weekend`, and `festival`.

A total of 17 raw representations were observed.

### Recommended normalization

1. Remove leading and trailing whitespace.
2. Convert all values to lowercase.
3. Map `fes.` to `festival`.
4. Map `wee.` to `weekend`.

### Normalized distribution

| Category | Count | Percentage |
|---|---:|---:|
| Weekend | 417 | 41.7% |
| Festival | 381 | 38.1% |
| Weekday | 202 | 20.2% |
| **Total** | **1,000** | **100%** |

After normalization, only three categories remain, with no missing or unmapped values.

---

## 10. Outlet-Level Analysis

The dataset contains three outlets.

| Outlet | Records | Unique Dates |
|---|---:|---:|
| College Gate | 476 | 175 |
| Mall Road | 266 | 138 |
| Station Road | 258 | 131 |
| **Total** | **1,000** | — |

### Monthly record distribution

| Month (2026) | College Gate | Mall Road | Station Road |
|---|---:|---:|---:|
| January | 76 | 41 | 37 |
| February | 74 | 43 | 45 |
| March | 90 | 55 | 41 |
| April | 81 | 48 | 42 |
| May | 84 | 40 | 41 |
| June | 71 | 39 | 52 |

Outlet names were found to be consistent, with no major whitespace or case-related issues.

---

## 11. Outlet-Date Structure

A total of 444 unique outlet-date combinations were observed.

- 305 outlet-date combinations contain multiple records.
- The maximum number of records for a single outlet-date combination is 9.

These multiple records may represent distinct observations rather than erroneous duplicates.

The exact row-level meaning of the dataset is not yet fully established. Therefore, records should not be aggregated by outlet and date or removed solely because multiple observations exist for the same outlet-date combination.

The row granularity should be clarified before aggregation or further restructuring.

---

## 12. Repeated Values Investigation

Some numerical features contain repeated values across observations.

For example, `competitor_shops` for College Gate remained equal to 2 across six observations between 30 January and 1 February.

Such repeated values may represent valid conditions rather than data entry problems. No automatic corrections are recommended based solely on repeated values.

---

## 13. Holdout Dataset

The file `holdout_features.csv` contains feature columns required for prediction but does not contain the target variable.

It also does not contain `loaves_in_closing_register`.

Therefore, preprocessing and feature engineering must ensure that the final model uses only features available in the holdout dataset.

The training, later, and holdout datasets are maintained using DVC for dataset versioning.

---

## 14. Proposed Data Cleaning Decisions

The following actions are recommended for the preprocessing stage:

| Issue | Proposed Action |
|---|---|
| Exact duplicate rows | Remove exact duplicates |
| Duplicate IDs after deduplication | Validate uniqueness |
| Date format | Convert to datetime |
| `day_type` inconsistencies | Normalize strings and map abbreviations |
| Suspected 10x target values | Flag and investigate; compare documented correction later |
| Closing register feature | Exclude from model features |
| Multiple records per outlet-date | Preserve until row granularity is understood |
| Repeated numerical values | Retain unless proven invalid |
| Raw datasets | Preserve unchanged |

All cleaning operations should be reproducible through scripts rather than performed manually on the raw datasets.

---

## 15. Conclusion

The dataset is generally complete, with no missing values and a continuous date range. However, several data quality issues require attention before model development:

1. Exact duplicate records.
2. Inconsistent categorical representations in `day_type`.
3. Fifteen suspected target scaling anomalies.
4. Multiple observations per outlet-date requiring clarification of row granularity.
5. A closing-register feature that is unavailable in the holdout dataset.

These issues have been identified and documented. The raw datasets will remain unchanged, and all cleaning decisions will be implemented reproducibly during the preprocessing stage.

**Stage 1 data investigation is complete. The next stage will focus on implementing and validating the data preprocessing pipeline.**
