"""Shared cleaning rules for the Titanic data.

Both notebooks import from here so the cleaning decisions live in one place.

The threshold rule from the brief:
    < 5% missing      -> drop those rows
    5% - 30% missing  -> impute
    > 30% missing     -> too high to impute reliably: drop the column or
                         treat "missing" as its own category
"""
import pandas as pd

DROP_ROWS_BELOW = 5.0   # percent
IMPUTE_UP_TO = 30.0     # percent


def missing_report(df: pd.DataFrame) -> pd.DataFrame:
    """Percent missing for every column that has any, with the rule it falls under."""
    pct = (df.isna().mean() * 100).round(2)
    pct = pct[pct > 0].sort_values(ascending=False)

    def rule(p):
        if p < DROP_ROWS_BELOW:
            return "< 5%  -> drop rows"
        if p <= IMPUTE_UP_TO:
            return "5-30% -> impute"
        return "> 30% -> own category / drop column"

    return pd.DataFrame({"missing_pct": pct, "strategy": pct.map(rule)})


def apply_row_rules(df: pd.DataFrame) -> pd.DataFrame:
    """Cleaning steps that don't learn anything from the data.

    Safe to run before a train/test split, because nothing here is
    computed from the values (no means, medians or modes):
      * embarked / embark_town (0.22% missing) -> drop those 2 rows
      * deck (77.22% missing) -> "Unknown" becomes its own category
    """
    out = df.dropna(subset=["embarked", "embark_town"]).copy()
    out["deck"] = out["deck"].astype("object").fillna("Unknown")
    return out.reset_index(drop=True)


def impute_age_by_group(df: pd.DataFrame) -> pd.DataFrame:
    """age (19.87% missing) -> median age of the same sex + passenger class.

    Used in the EDA notebook only. The modeling notebook imputes age inside
    its Pipeline instead, using medians learned from the training split only.
    """
    out = df.copy()
    group_median = out.groupby(["sex", "pclass"])["age"].transform("median")
    out["age"] = out["age"].fillna(group_median)
    return out
