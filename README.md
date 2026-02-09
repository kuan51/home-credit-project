# Home Credit Default Risk

Predicting loan defaults for underserved borrowers using the [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk) Kaggle dataset. This project follows the CRISP-DM (Cross-Industry Standard Process for Data Mining) framework.

**Author:** Rex Linder | **Course:** IS 6850-004 Predictive Analytics Project (Spring 2026) | University of Utah MSBA Capstone

---

## 1. Business Understanding

Home Credit is an international consumer finance provider focused on lending to consumers with little or no credit history. They need better credit risk models to expand access to underserved populations while managing default risk.

**Success Criteria**

- Reduce loan defaults by at least 10% through improved credit decision accuracy
- Maintain or increase approval volume for creditworthy applicants
- Reduce cost of capital through better risk provisioning

**Approach**

Build a supervised binary classification model on historical loan data to estimate probability of default. Results are delivered as a risk score that integrates into existing approval workflows.

**Scope**

Data preparation, model development, and performance analysis. Production deployment is out of scope for this phase.

## 2. Data Understanding

The dataset consists of 8 related CSV tables joined by `SK_ID_CURR`:

| Table | Records | Features | Description |
|---|---:|---:|---|
| application_train | 307,511 | 122 | Main application data with TARGET |
| application_test | 48,744 | 121 | Test set for Kaggle submission |
| bureau | 1.7M | 17 | Credit bureau history |
| bureau_balance | 27.3M | 3 | Monthly bureau balance |
| credit_card_balance | 3.8M | 23 | Credit card snapshots |
| installments_payments | 13.6M | 8 | Payment history |
| POS_CASH_balance | 10.0M | 8 | POS/cash loan snapshots |
| previous_application | 1.7M | 37 | Previous Home Credit applications |

**Key EDA Findings**

- **Class imbalance:** ~8% default rate requires careful handling (class weights, AUC-ROC evaluation)
- **Strongest predictors:** EXT_SOURCE scores (external credit bureau) show clear separation between defaulters and non-defaulters
- **Demographic signals:** Income type, education level, and age correlate with default rate
- **Data quality issues:**
  - `DAYS_EMPLOYED = 365243` placeholder in ~55K rows (unemployed/pensioners)
  - Housing columns have >50% missing values
  - `CODE_GENDER` contains 4 XNA values

See [`EDA.html`](EDA.html) for the full exploratory analysis.

## 3. Data Preparation

`data_preparation.R` implements an 18-function modular pipeline that transforms raw data into analysis-ready features.

**Pipeline Steps**

1. **Clean application data** -- Fix DAYS_EMPLOYED placeholder, CODE_GENDER XNA, drop high-missingness columns, impute missing numerics (median), cap income outliers (99th percentile), one-hot encode categoricals
2. **Engineer features** -- Age/employment in years, financial ratios (credit-to-income, annuity-to-income), missing-value flags, binned age groups, averaged EXT_SOURCE score
3. **Aggregate supplementary tables** -- Summarize bureau history (credit counts, debt ratios), previous applications (approval rates), and installment payments (late payment rates, recent trends) to one row per client
4. **Join and align** -- Left-join aggregated features, fill NAs with 0, verify train/test column parity

**Train/Test Consistency**

All thresholds (medians, percentile caps, category levels, columns to drop) are computed from training data only and applied identically to both sets.

**Output**

| Dataset | Rows | Columns |
|---|---:|---:|
| train_prepared.csv | 307,511 | 149 |
| test_prepared.csv | 48,744 | 148 |

```r
source("data_preparation.R")
result <- run_pipeline()
```

## 4. Modeling

*In progress.*

## 5. Evaluation

*Upcoming. Primary metric: AUC-ROC.*

## 6. Deployment

Out of scope for this phase. Future work may include integration testing and production deployment.

---

## Project Structure

```
README.md                        # This file
EDA.qmd                         # Exploratory data analysis (Quarto)
EDA.html                        # Rendered EDA report
data_preparation.R              # Data cleaning & feature engineering pipeline
Business Problem Statement.docx # Project scope document
data/                           # Raw CSV data (~3.7GB, gitignored)
```

## Tech Stack

R, tidyverse, Quarto, patchwork, knitr

```bash
# Install R dependencies
Rscript -e 'install.packages(c("tidyverse", "patchwork", "knitr"))'

# Render EDA report
quarto render EDA.qmd

# Run data preparation pipeline
Rscript -e 'source("data_preparation.R"); run_pipeline()'
```

## AI Use Disclosure

- Claude Code was used for code generation assistance, code review, and debugging during EDA and data preparation development
- ChatGPT was used to review EDA outline structure and headings
