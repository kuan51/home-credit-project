# =========================================
# data_preparation.R
# Home Credit Default Risk - Data Preparation
# =========================================
# This script cleans, transforms, and engineers features from the
# Home Credit dataset. It follows the EDA recommendations and produces
# analysis-ready train/test CSV files.
#
# Usage:
#   source("data_preparation.R")
#   result <- run_pipeline()
# =========================================

library(tidyverse)

# =========================================
# SECTION 1: Clean Application Data
# =========================================
# These functions fix data quality issues identified in the EDA:
#   - DAYS_EMPLOYED placeholder (365243)
#   - CODE_GENDER "XNA" values
#   - High-missingness housing columns (>50% NA)
#   - Missing numeric values (EXT_SOURCE, AMT_ANNUITY, AMT_GOODS_PRICE)
#   - Extreme outliers in AMT_INCOME_TOTAL
#   - Categorical encoding for low-cardinality columns

# --- Function 1: fix_days_employed ---
# The EDA found ~55,000 rows where DAYS_EMPLOYED = 365243.
# This is a placeholder meaning "unemployed or pensioner."
# We replace it with NA and create a binary flag to preserve this info.
fix_days_employed <- function(df) {
  df <- df |>
    mutate(
      # 1 = was placeholder (unemployed/pensioner), 0 = normal employment data
      FLAG_EMPLOYED_PLACEHOLDER = if_else(DAYS_EMPLOYED == 365243, 1, 0),
      # Replace placeholder with NA so it doesn't skew calculations
      DAYS_EMPLOYED = if_else(DAYS_EMPLOYED == 365243, NA_real_, DAYS_EMPLOYED)
    )
  return(df)
}

# --- Function 2: fix_code_gender ---
# The EDA found 4 rows with CODE_GENDER = "XNA" (unknown gender).
# We replace "XNA" with NA so R treats it as missing data.
fix_code_gender <- function(df) {
  df <- df |>
    mutate(
      CODE_GENDER = if_else(CODE_GENDER == "XNA", NA_character_, CODE_GENDER)
    )
  return(df)
}

# --- Function 3: drop_high_missing_cols ---
# The EDA found many housing columns (COMMONAREA_*, LANDAREA_*, etc.)
# have >50% missing values. Columns with this much missing data
# have limited predictive value, so we drop them.
#
# For train/test consistency:
#   - On training data: pass cols_to_drop = NULL to compute from the data
#   - On test data: pass the cols_to_drop list computed from training
#
# Returns a list with two elements:
#   $df           = the dataframe with high-missing columns removed
#   $cols_to_drop = the column names that were dropped (save for test data)
drop_high_missing_cols <- function(df, threshold = 0.5, cols_to_drop = NULL) {
  # If no list provided, compute which columns to drop (training mode)
  if (is.null(cols_to_drop)) {
    na_fractions <- colMeans(is.na(df))
    cols_to_drop <- names(na_fractions[na_fractions > threshold])
    # Don't accidentally drop TARGET if it happens to have NAs
    cols_to_drop <- setdiff(cols_to_drop, "TARGET")
  }

  if (length(cols_to_drop) > 0) {
    cat("Dropping", length(cols_to_drop), "columns with >",
        threshold * 100, "% missing:\n")
    cat(" ", paste(cols_to_drop, collapse = ", "), "\n")
  }

  # Only drop columns that exist in this dataframe
  cols_present <- intersect(cols_to_drop, names(df))
  df <- df |> select(-all_of(cols_present))

  return(list(df = df, cols_to_drop = cols_to_drop))
}

# --- Function 4: impute_numeric_missing ---
# Fill missing values in key numeric columns with their median.
# EDA recommended median imputation for columns with <50% missing:
#   EXT_SOURCE_1, EXT_SOURCE_2, EXT_SOURCE_3, AMT_ANNUITY, AMT_GOODS_PRICE
#
# For train/test consistency:
#   - On training data: pass medians = NULL to compute medians from the data
#   - On test data: pass the medians computed from training data
#
# Returns a list with two elements:
#   $df      = the dataframe with NAs filled
#   $medians = the median values used (save these for test data)
impute_numeric_missing <- function(df, medians = NULL) {
  # Columns to impute
  cols_to_impute <- c("EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3",
                      "AMT_ANNUITY", "AMT_GOODS_PRICE")

  # Only impute columns that actually exist in the dataframe
  cols_to_impute <- intersect(cols_to_impute, names(df))

  # If no medians provided, compute them from this data (training mode)
  if (is.null(medians)) {
    medians <- list()
    for (col in cols_to_impute) {
      medians[[col]] <- median(df[[col]], na.rm = TRUE)
    }
  }

  # Replace NAs with the corresponding median
  for (col in cols_to_impute) {
    df[[col]] <- if_else(is.na(df[[col]]), medians[[col]], df[[col]])
  }

  return(list(df = df, medians = medians))
}

# --- Function 5: cap_income_outliers ---
# The EDA found extreme outliers in AMT_INCOME_TOTAL (up to 117 million).
# We cap values at the 99th percentile to reduce outlier influence.
#
# For train/test consistency:
#   - On training data: pass cap_value = NULL to compute from the data
#   - On test data: pass the cap_value computed from training data
#
# Returns a list with two elements:
#   $df        = the dataframe with capped income
#   $cap_value = the threshold used (save this for test data)
cap_income_outliers <- function(df, cap_value = NULL) {
  # If no cap provided, compute 99th percentile (training mode)
  if (is.null(cap_value)) {
    cap_value <- quantile(df$AMT_INCOME_TOTAL, 0.99, na.rm = TRUE)
    cat("Income cap set at 99th percentile:", cap_value, "\n")
  }

  # pmin() caps each value: keeps the smaller of the value or the cap
  df <- df |>
    mutate(
      AMT_INCOME_TOTAL = pmin(AMT_INCOME_TOTAL, cap_value)
    )

  return(list(df = df, cap_value = cap_value))
}

# --- Function 6: encode_categoricals ---
# The EDA recommended one-hot encoding for low-cardinality categorical columns.
# This converts categories like "M"/"F" into separate binary (0/1) columns.
# We encode: CODE_GENDER, FLAG_OWN_CAR, FLAG_OWN_REALTY,
#            NAME_EDUCATION_TYPE, NAME_FAMILY_STATUS, NAME_INCOME_TYPE
#
# For train/test consistency:
#   - On training data: pass cat_levels = NULL to learn categories from the data
#   - On test data: pass the cat_levels computed from training
#     (ensures test gets the same columns, even if some categories are absent)
#
# Returns a list with two elements:
#   $df         = the dataframe with one-hot encoded columns
#   $cat_levels = the category levels used (save for test data)
encode_categoricals <- function(df, cat_levels = NULL) {
  # Drop high-cardinality categoricals (too many levels for one-hot encoding)
  high_card_cols <- c("OCCUPATION_TYPE", "ORGANIZATION_TYPE")
  cols_to_remove <- intersect(high_card_cols, names(df))
  if (length(cols_to_remove) > 0) {
    cat("Dropping high-cardinality categoricals:", paste(cols_to_remove, collapse = ", "), "\n")
    df <- df |> select(-all_of(cols_to_remove))
  }

  # One-hot encode low-cardinality categoricals (full-rank; drop-first if using linear models)
  cat_cols <- c("CODE_GENDER", "FLAG_OWN_CAR", "FLAG_OWN_REALTY",
                "NAME_EDUCATION_TYPE", "NAME_FAMILY_STATUS", "NAME_INCOME_TYPE",
                "NAME_CONTRACT_TYPE", "NAME_TYPE_SUITE", "NAME_HOUSING_TYPE",
                "WEEKDAY_APPR_PROCESS_START", "EMERGENCYSTATE_MODE")

  # Only encode columns that exist in the dataframe
  cat_cols <- intersect(cat_cols, names(df))

  # If no levels provided, learn them from this data (training mode)
  if (is.null(cat_levels)) {
    cat_levels <- list()
    for (col in cat_cols) {
      cat_levels[[col]] <- sort(na.omit(unique(df[[col]])))
    }
  }

  for (col in cat_cols) {
    # Use the levels from cat_levels (same for train and test)
    for (val in cat_levels[[col]]) {
      # Build the new column name: original name + underscore + value
      # Clean the value: replace spaces with underscores, remove special chars
      clean_val <- str_replace_all(val, "[^A-Za-z0-9]", "_")
      new_col <- paste0(col, "_", clean_val)
      df[[new_col]] <- if_else(df[[col]] == val, 1, 0, missing = 0)
    }
  }

  # Remove the original categorical columns (replaced by binary columns)
  df <- df |> select(-all_of(cat_cols))

  return(list(df = df, cat_levels = cat_levels))
}

# =========================================
# SECTION 2: Engineer Features
# =========================================
# These functions create new columns derived from existing data.
# Feature engineering often improves model performance by making
# patterns more explicit for the algorithm.

# --- Function 7: add_time_features ---
# DAYS_BIRTH and DAYS_EMPLOYED are negative numbers (days before application).
# We convert them to positive years for easier interpretation.
# Example: DAYS_BIRTH = -10000 means born ~27.4 years before applying.
add_time_features <- function(df) {
  df <- df |>
    mutate(
      # Divide by -365.25 to convert negative days to positive years
      # (365.25 accounts for leap years)
      AGE_YEARS = -DAYS_BIRTH / 365.25,
      # DAYS_EMPLOYED may be NA (after fixing the 365243 placeholder)
      EMPLOYED_YEARS = if_else(
        is.na(DAYS_EMPLOYED), NA_real_, -DAYS_EMPLOYED / 365.25
      )
    )
  return(df)
}

# --- Function 8: add_financial_ratios ---
# Financial ratios capture how much of a client's income goes to loan payments.
# These are commonly used in credit risk modeling:
#   - credit_to_income: How large is the loan relative to yearly income?
#   - annuity_to_income: What fraction of income goes to loan payments?
#   - credit_to_annuity: How many payment periods to repay? (loan duration proxy)
#   - credit_to_goods: Is the loan larger than the goods purchased? (over-borrowing)
add_financial_ratios <- function(df) {
  df <- df |>
    mutate(
      # pmax(..., 1) prevents division by zero
      credit_to_income  = AMT_CREDIT / pmax(AMT_INCOME_TOTAL, 1),
      annuity_to_income = AMT_ANNUITY / pmax(AMT_INCOME_TOTAL, 1),
      credit_to_annuity = AMT_CREDIT / pmax(AMT_ANNUITY, 1),
      credit_to_goods   = AMT_CREDIT / pmax(AMT_GOODS_PRICE, 1)
    )
  return(df)
}

# --- Function 9: add_missing_flags ---
# Missing data can itself be informative. For example, a client with no
# external credit score may be higher risk. We create binary flags
# (1 = was missing, 0 = had a value) for key columns.
#
# IMPORTANT: This function must be called BEFORE impute_numeric_missing(),
# because after imputation the NAs are filled and we'd lose this information.
add_missing_flags <- function(df) {
  df <- df |>
    mutate(
      FLAG_EXT_SOURCE_1_MISSING    = if_else(is.na(EXT_SOURCE_1), 1, 0),
      FLAG_EXT_SOURCE_2_MISSING    = if_else(is.na(EXT_SOURCE_2), 1, 0),
      FLAG_EXT_SOURCE_3_MISSING    = if_else(is.na(EXT_SOURCE_3), 1, 0),
      FLAG_OCCUPATION_TYPE_MISSING = if_else(is.na(OCCUPATION_TYPE), 1, 0)
    )
  return(df)
}

# --- Function 10: add_binned_and_interaction ---
# Binned variables group continuous values into categories.
# Interaction features combine multiple columns into one.
#   - AGE_GROUP: Bins age into interpretable life-stage groups
#   - EXT_SOURCE_AVG: Average of the 3 external credit scores
#     (the EDA found all 3 are predictive; averaging smooths out noise)
add_binned_and_interaction <- function(df) {
  # Fixed age bins based on common life stages
  age_breaks <- c(0, 25, 35, 45, 55, 65, Inf)
  age_labels <- c("18-25", "26-35", "36-45", "46-55", "56-65", "65+")

  # One-hot encode age bins directly (cut() returns a factor, which we avoid keeping)
  age_group <- cut(df$AGE_YEARS,
                   breaks = age_breaks,
                   labels = age_labels,
                   include.lowest = TRUE,
                   right = TRUE)

  # Create one binary column per age bin
  for (label in age_labels) {
    col_name <- paste0("AGE_GROUP_", gsub("-", "_", label))
    df[[col_name]] <- if_else(age_group == label, 1, 0, missing = 0)
  }

  # Average of the three external scores (already imputed by this point)
  df <- df |>
    mutate(
      EXT_SOURCE_AVG = (EXT_SOURCE_1 + EXT_SOURCE_2 + EXT_SOURCE_3) / 3
    )
  return(df)
}

# =========================================
# SECTION 3: Aggregate Supplementary Tables
# =========================================
# The supplementary tables have MANY rows per client (e.g., one row per
# previous credit in bureau.csv). We need to aggregate them down to
# ONE row per client (SK_ID_CURR) with summary statistics.
# These aggregated features capture a client's credit history.

# --- Function 11: aggregate_bureau ---
# bureau.csv has one row per credit reported to the credit bureau.
# We summarize each client's bureau history:
#   - How many credits do they have?
#   - How many are active vs closed?
#   - How much is overdue?
#   - What's their average debt ratio (debt / total credit)?
aggregate_bureau <- function(bureau_path = "./data/bureau.csv") {
  cat("Reading bureau.csv...\n")
  bureau <- read_csv(bureau_path, show_col_types = FALSE)

  bureau_agg <- bureau |>
    group_by(SK_ID_CURR) |>
    summarise(
      # Total number of credits on record
      bureau_credit_count = n(),
      # Count of currently active credits
      bureau_active_count = sum(CREDIT_ACTIVE == "Active", na.rm = TRUE),
      # Count of closed credits
      bureau_closed_count = sum(CREDIT_ACTIVE == "Closed", na.rm = TRUE),
      # Total overdue amount across all credits
      bureau_overdue_total = sum(AMT_CREDIT_SUM_OVERDUE, na.rm = TRUE),
      # Average debt ratio: how much of each credit is still owed
      # pmax(..., 1) prevents division by zero
      bureau_debt_ratio_avg = mean(
        AMT_CREDIT_SUM_DEBT / pmax(AMT_CREDIT_SUM, 1), na.rm = TRUE
      ),
      .groups = "drop"
    )

  cat("Bureau aggregated:", nrow(bureau_agg), "clients\n")
  return(bureau_agg)
}

# --- Function 12: aggregate_previous ---
# previous_application.csv has one row per previous Home Credit application.
# We summarize each client's application history:
#   - How many times did they apply?
#   - How many were approved vs refused?
#   - What's their approval rate?
aggregate_previous <- function(prev_path = "./data/previous_application.csv") {
  cat("Reading previous_application.csv...\n")
  prev <- read_csv(prev_path, show_col_types = FALSE)

  prev_agg <- prev |>
    group_by(SK_ID_CURR) |>
    summarise(
      # Total number of previous applications
      prev_app_count = n(),
      # Count approved applications
      prev_approved_count = sum(NAME_CONTRACT_STATUS == "Approved", na.rm = TRUE),
      # Count refused applications
      prev_refused_count = sum(NAME_CONTRACT_STATUS == "Refused", na.rm = TRUE),
      # Approval rate: fraction of applications that were approved
      prev_approval_rate = sum(NAME_CONTRACT_STATUS == "Approved", na.rm = TRUE) / n(),
      .groups = "drop"
    )

  cat("Previous applications aggregated:", nrow(prev_agg), "clients\n")
  return(prev_agg)
}

# --- Function 13: aggregate_installments ---
# installments_payments.csv has one row per payment on a previous loan.
# This is the largest file (13.6M rows), so we only load needed columns.
# We summarize each client's payment behavior:
#   - How many payments did they make?
#   - What fraction were late?
#   - On average, did they pay the full amount or more/less?
aggregate_installments <- function(inst_path = "./data/installments_payments.csv") {
  cat("Reading installments_payments.csv (this may take a moment)...\n")

  # Only load the columns we need to save memory
  inst <- read_csv(inst_path, show_col_types = FALSE,
                   col_select = c(SK_ID_CURR, DAYS_INSTALMENT, DAYS_ENTRY_PAYMENT,
                                  AMT_INSTALMENT, AMT_PAYMENT))

  inst_agg <- inst |>
    mutate(
      # A payment is "late" if it was made after the due date.
      # Both columns are negative (days before application).
      # Less negative = more recent. So if DAYS_ENTRY_PAYMENT > DAYS_INSTALMENT,
      # the payment happened after the due date = late.
      is_late = if_else(DAYS_ENTRY_PAYMENT > DAYS_INSTALMENT, 1, 0),
      # Ratio of amount paid to amount owed
      # > 1 means overpaid, < 1 means underpaid
      payment_ratio = AMT_PAYMENT / pmax(AMT_INSTALMENT, 1)
    ) |>
    group_by(SK_ID_CURR) |>
    summarise(
      # Total number of installment payments
      inst_total_count = n(),
      # Fraction of payments that were late (0 to 1)
      inst_late_pct = mean(is_late, na.rm = TRUE),
      # Average payment ratio (1.0 = exactly on target)
      inst_payment_ratio_avg = mean(payment_ratio, na.rm = TRUE),
      # Late-payment rate in the most recent 25% of installments (payment trend)
      inst_late_pct_recent = mean(
        is_late[DAYS_INSTALMENT >= quantile(DAYS_INSTALMENT, 0.75, na.rm = TRUE)],
        na.rm = TRUE
      ),
      .groups = "drop"
    )

  cat("Installments aggregated:", nrow(inst_agg), "clients\n")
  return(inst_agg)
}

# =========================================
# SECTION 4: Join Aggregated Features
# =========================================
# Combine the aggregated supplementary data with the application data.
# We use left joins so every applicant keeps their row, even if they
# have no history in a supplementary table (those get NAs, then filled with 0).

# --- Function 14: join_all_features ---
# Left-join all three aggregated tables to the application data.
# Clients with no history in a table will get NA values from the join,
# which we replace with 0 (no history = zero counts/rates).
join_all_features <- function(app_df, bureau_agg, prev_agg, inst_agg) {
  # Join all supplementary tables by client ID
  result <- app_df |>
    left_join(bureau_agg, by = "SK_ID_CURR") |>
    left_join(prev_agg,   by = "SK_ID_CURR") |>
    left_join(inst_agg,   by = "SK_ID_CURR")

  # List of all columns that came from the aggregated tables
  agg_cols <- c(
    "bureau_credit_count", "bureau_active_count", "bureau_closed_count",
    "bureau_overdue_total", "bureau_debt_ratio_avg",
    "prev_app_count", "prev_approved_count", "prev_refused_count",
    "prev_approval_rate",
    "inst_total_count", "inst_late_pct", "inst_payment_ratio_avg",
    "inst_late_pct_recent"
  )

  # Only fill columns that actually exist (in case a join was skipped)
  agg_cols <- intersect(agg_cols, names(result))

  # Replace NAs with 0: no prior history means zero counts/rates
  result <- result |>
    mutate(across(all_of(agg_cols), ~replace_na(., 0)))

  return(result)
}

# =========================================
# SECTION 5: Pipeline Orchestration
# =========================================
# These functions tie everything together. The key idea is:
#   1. Compute thresholds/medians from TRAINING data only
#   2. Apply the same thresholds to both train and test
#   3. Verify that train and test end up with the same columns

# --- Function 15: compute_train_params ---
# Compute all data-dependent values from the raw training data.
# These values are stored in a plain list and reused when processing test data.
# This ensures train and test are transformed identically.
compute_train_params <- function(train_df) {
  # Columns to compute medians for
  median_cols <- c("EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3",
                   "AMT_ANNUITY", "AMT_GOODS_PRICE")

  # Compute median for each column
  medians <- list()
  for (col in median_cols) {
    medians[[col]] <- median(train_df[[col]], na.rm = TRUE)
  }

  # Compute 99th percentile for income capping
  income_cap <- quantile(train_df$AMT_INCOME_TOTAL, 0.99, na.rm = TRUE)

  params <- list(
    medians    = medians,
    income_cap = income_cap
  )

  cat("Training parameters computed:\n")
  cat("  Medians:", paste(names(medians), round(unlist(medians), 2),
                          sep = "=", collapse = ", "), "\n")
  cat("  Income cap (99th pctl):", income_cap, "\n")

  return(params)
}

# --- Function 16: prepare_application ---
# Full cleaning + feature engineering pipeline for ONE dataset.
# Call this on both train and test, passing the SAME params to both.
#
# The order of operations matters:
#   1. Missing flags FIRST (before imputation fills in the NAs)
#   2. Cleaning (fix placeholders, impute, drop cols, cap)
#   3. Encoding (one-hot categoricals)
#   4. Feature engineering (time, ratios, bins)
#
# Returns a list with two elements:
#   $df     = the cleaned and engineered dataframe
#   $params = updated params (with cols_to_drop and cat_levels added)
prepare_application <- function(df, params) {
  # Step 1: Create missing flags BEFORE imputation
  df <- add_missing_flags(df)

  # Step 2: Fix known data quality issues
  df <- fix_days_employed(df)
  df <- fix_code_gender(df)

  # Step 3: Impute missing values BEFORE dropping columns
  # (EXT_SOURCE_1 has >50% missing -- impute first so it doesn't get dropped)
  impute_result <- impute_numeric_missing(df, medians = params$medians)
  df <- impute_result$df

  # Step 4: Drop columns with >50% missing (housing columns)
  # EXT_SOURCE columns are now filled, so only housing columns get dropped
  # On training: computes which columns to drop
  # On test: uses the same list from training (params$cols_to_drop)
  drop_result <- drop_high_missing_cols(df, cols_to_drop = params$cols_to_drop)
  df <- drop_result$df

  # Step 5: Cap income outliers at training 99th percentile
  cap_result <- cap_income_outliers(df, cap_value = params$income_cap)
  df <- cap_result$df

  # Step 6: One-hot encode categorical columns
  # On training: learns the category levels
  # On test: uses the same levels from training (params$cat_levels)
  encode_result <- encode_categoricals(df, cat_levels = params$cat_levels)
  df <- encode_result$df

  # Step 7: Create derived features
  df <- add_time_features(df)
  df <- add_financial_ratios(df)
  df <- add_binned_and_interaction(df)

  # Save the computed values back into params for reuse on test data
  params$cols_to_drop <- drop_result$cols_to_drop
  params$cat_levels   <- encode_result$cat_levels

  return(list(df = df, params = params))
}

# --- Function 17: align_columns ---
# Verify that train and test have identical columns (except TARGET).
# This catches any mismatch that could break modeling later.
align_columns <- function(train, test) {
  train_cols <- sort(setdiff(names(train), "TARGET"))
  test_cols  <- sort(names(test))

  if (identical(train_cols, test_cols)) {
    cat("Column check PASSED: train and test have identical features.\n")
  } else {
    in_train_only <- setdiff(train_cols, test_cols)
    in_test_only  <- setdiff(test_cols, train_cols)
    msg <- "Column MISMATCH between train and test!\n"
    if (length(in_train_only) > 0) {
      msg <- paste0(msg, "  In train only: ", paste(in_train_only, collapse = ", "), "\n")
    }
    if (length(in_test_only) > 0) {
      msg <- paste0(msg, "  In test only: ", paste(in_test_only, collapse = ", "), "\n")
    }
    stop(msg)
  }
}

# --- Function 18: run_pipeline ---
# Main entry point. Loads data, runs all steps, saves output.
# Usage: result <- run_pipeline()
run_pipeline <- function() {
  # ---- Load application data ----
  cat("=== Loading application data ===\n")
  train_raw <- read_csv("./data/application_train.csv", show_col_types = FALSE)
  test_raw  <- read_csv("./data/application_test.csv",  show_col_types = FALSE)
  cat("Train:", nrow(train_raw), "rows x", ncol(train_raw), "cols\n")
  cat("Test: ", nrow(test_raw),  "rows x", ncol(test_raw),  "cols\n\n")

  # ---- Step 1: Compute parameters from training data ----
  cat("=== Computing training parameters ===\n")
  params <- compute_train_params(train_raw)
  cat("\n")

  # ---- Step 2: Aggregate supplementary tables ----
  cat("=== Aggregating supplementary tables ===\n")
  bureau_agg <- aggregate_bureau()
  prev_agg   <- aggregate_previous()
  inst_agg   <- aggregate_installments()
  cat("\n")

  # ---- Step 3: Clean and engineer features ----
  cat("=== Preparing training data ===\n")
  train_result <- prepare_application(train_raw, params)
  train  <- train_result$df
  params <- train_result$params  # now includes cols_to_drop and cat_levels

  cat("\n=== Preparing test data ===\n")
  test_result <- prepare_application(test_raw, params)
  test <- test_result$df
  cat("\n")

  # ---- Step 4: Join aggregated features ----
  cat("=== Joining aggregated features ===\n")
  train <- join_all_features(train, bureau_agg, prev_agg, inst_agg)
  test  <- join_all_features(test,  bureau_agg, prev_agg, inst_agg)
  cat("\n")

  # ---- Step 5: Verify column alignment ----
  cat("=== Verifying train/test consistency ===\n")
  align_columns(train, test)
  cat("\n")

  # ---- Step 6: Save outputs ----
  cat("=== Saving prepared data ===\n")
  write_csv(train, "./data/train_prepared.csv")
  write_csv(test,  "./data/test_prepared.csv")
  cat("Saved: ./data/train_prepared.csv\n")
  cat("Saved: ./data/test_prepared.csv\n\n")

  cat("=== DONE ===\n")
  cat("Training set:", nrow(train), "rows x", ncol(train), "columns\n")
  cat("Test set:    ", nrow(test),  "rows x", ncol(test),  "columns\n")

  # Return both datasets (useful if running interactively)
  return(list(train = train, test = test, params = params))
}
