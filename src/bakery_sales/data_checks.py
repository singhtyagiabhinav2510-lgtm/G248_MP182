"""
Data Quality and Integrity Checks for MP182 (Jaipur Bakery Bread Sales).
Verifies the seven data problem categories outlined in the Student Handbook (Section 7).
Run directly with:
    python -m src.bakery_sales.data_checks
"""

from pathlib import Path
import numpy as np
import pandas as pd


def check_leaking_column(df_train: pd.DataFrame, df_holdout: pd.DataFrame) -> dict:
    """
    Checks for columns present in train.csv but missing in holdout_features.csv,
    which secretly contain target information or post-event closing records.
    """
    train_cols = set(df_train.columns)
    holdout_cols = set(df_holdout.columns)
    missing_in_holdout = train_cols - holdout_cols - {"loaves_sold"}

    if "loaves_in_closing_register" in missing_in_holdout:
        corr = df_train["loaves_in_closing_register"].corr(df_train["loaves_sold"])
        return {
            "present": True,
            "column": "loaves_in_closing_register",
            "details": f"Missing from holdout; records closing stock register (correlation: {corr:.3f})",
        }
    return {"present": False, "column": None}


def check_unit_mixup(df: pd.DataFrame, group_col: str = "outlet") -> dict:
    """
    Checks if any numeric feature in one group was recorded in a different unit
    (e.g., 10x, 100x, 1000x difference between groups).
    """
    numeric_cols = ["price_rs", "temp_c", "footfall_index", "competitor_shops", "shelf_hours", "orders_previous_day"]
    for col in numeric_cols:
        means = df.groupby(group_col)[col].mean()
        min_mean = means.min()
        max_mean = means.max()
        if min_mean > 0 and (max_mean / min_mean) > 5.0:
            return {
                "present": True,
                "column": col,
                "group_value": str(means.idxmax()),
                "factor": float(round(max_mean / min_mean)),
            }
    return {"present": False, "column": None, "group_value": None, "factor": None}


def check_sentinel_values(df: pd.DataFrame) -> dict:
    """
    Checks for special placeholder values (-1, -999, 999, 9999) used instead of blanks.
    """
    numeric_cols = ["price_rs", "temp_c", "footfall_index", "competitor_shops", "shelf_hours", "orders_previous_day"]
    sentinels = [-9999, -999, -1, 999, 9999]
    for col in numeric_cols:
        vc = df[col].value_counts()
        for sentinel in sentinels:
            if sentinel in vc.index and vc[sentinel] > 10:
                return {
                    "present": True,
                    "column": col,
                    "value": sentinel,
                    "count": int(vc[sentinel]),
                }
    return {"present": False, "column": None, "value": None, "count": None}


def check_duplicate_rows(df: pd.DataFrame) -> dict:
    """
    Counts exact duplicate rows in the training dataset.
    """
    extra_copies = int(df.duplicated().sum())
    return {
        "present": extra_copies > 0,
        "count": extra_copies,
    }


def check_stuck_sensor(df: pd.DataFrame, date_col: str = "date", threshold_days: int = 14) -> dict:
    """
    Checks if any column repeated the exact same reading continuously for many days.
    """
    df_sorted = df.sort_values(date_col).copy()
    numeric_cols = ["temp_c", "footfall_index"]
    for col in numeric_cols:
        runs = (df_sorted[col] != df_sorted[col].shift(1)).cumsum()
        counts = df_sorted.groupby(runs)[col].count()
        if counts.max() >= threshold_days:
            top_run_id = counts.idxmax()
            subset = df_sorted[runs == top_run_id]
            return {
                "present": True,
                "column": col,
                "start_date": str(subset[date_col].min()),
                "end_date": str(subset[date_col].max()),
            }
    return {"present": False, "column": None, "start_date": None, "end_date": None}


def check_target_typos(df: pd.DataFrame, target_col: str = "loaves_sold", threshold: float = 1000.0) -> dict:
    """
    Checks for targets with an extra zero (10x scaling typo).
    """
    mask = df[target_col] >= threshold
    count = int(mask.sum())
    return {
        "present": count > 0,
        "count": count,
    }


def check_spelling_variants(df: pd.DataFrame, col: str = "day_type") -> dict:
    """
    Checks categorical columns for case, trailing space, or abbreviation variants.
    """
    raw_unique = df[col].dropna().unique()
    normalized = df[col].astype(str).str.strip().str.lower().replace({"fes.": "festival", "wee.": "weekend"})
    true_levels = int(normalized.nunique())
    has_variants = len(raw_unique) > true_levels
    return {
        "present": has_variants,
        "column": col,
        "raw_categories_count": len(raw_unique),
        "true_levels": true_levels,
    }


def run_all_checks(train_path: str = "data/raw/train.csv", holdout_path: str = "data/raw/holdout_features.csv") -> dict:
    root = Path(__file__).resolve().parents[2]
    df_train = pd.read_csv(root / train_path)
    df_holdout = pd.read_csv(root / holdout_path)

    results = {
        "leaking_column": check_leaking_column(df_train, df_holdout),
        "unit_mixup": check_unit_mixup(df_train),
        "sentinel_value": check_sentinel_values(df_train),
        "duplicate_rows": check_duplicate_rows(df_train),
        "stuck_sensor": check_stuck_sensor(df_train),
        "target_typos": check_target_typos(df_train),
        "spelling_variants": check_spelling_variants(df_train),
    }

    print("=" * 65)
    print("MP182 DATA QUALITY AUDIT RESULTS (7 HANDBOOK CATEGORIES)")
    print("=" * 65)
    for check_name, data in results.items():
        present_str = "[FOUND]" if data.get("present") else "[CLEAN]"
        print(f"{present_str} {check_name:<20}: {data}")
    print("=" * 65)

    return results


if __name__ == "__main__":
    run_all_checks()
