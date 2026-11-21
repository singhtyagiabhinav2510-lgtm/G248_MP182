import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def clean_data(df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
    """
    Cleans raw bakery dataset according to documented data quality findings:
    - Removes duplicate rows
    - Normalizes day_type strings and abbreviations
    - Corrects 10x target typographical errors (loaves_sold >= 1000)
    - Drops data leakage column 'loaves_in_closing_register' if present
    """
    df_clean = df.copy()

    # 1. Remove duplicate rows
    df_clean = df_clean.drop_duplicates()

    # 2. Normalize day_type
    if "day_type" in df_clean.columns:
        df_clean["day_type"] = df_clean["day_type"].astype(str).str.strip().str.lower()
        mapping = {
            "fes.": "festival",
            "wee.": "weekend",
        }
        df_clean["day_type"] = df_clean["day_type"].replace(mapping)

    # 3. Correct target typo (10x scaling error)
    if is_training and "loaves_sold" in df_clean.columns:
        mask = df_clean["loaves_sold"] >= 1000
        df_clean.loc[mask, "loaves_sold"] = df_clean.loc[mask, "loaves_sold"] / 10.0

    # 4. Remove data-leakage column if present
    if "loaves_in_closing_register" in df_clean.columns:
        df_clean = df_clean.drop(columns=["loaves_in_closing_register"])

    return df_clean

def get_preprocessor(num_features: list[str], cat_features: list[str]) -> ColumnTransformer:
    """
    Returns a ColumnTransformer that scales numerical features
    and one-hot encodes categorical features.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_features),
        ],
        remainder="drop",
    )
    return preprocessor