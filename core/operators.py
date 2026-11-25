"""Operator registry + tabular cleaning operators.

The LLM never writes cleaning code. It only *selects* operators from this
registry and *configures* their params. Every operator here is audited,
deterministic, and safe to run on untrusted config (but not untrusted code).

An operator is a function `op(df, **params) -> df`. It is registered with a
param schema (name -> allowed values / type hint) and a one-line description.
The schema doubles as the catalog we show the planner LLM, and as a validator
for configs coming back from the LLM.
"""

import numpy as np
import pandas as pd

# name -> {"fn", "params", "desc"}
OPERATORS = {}


def operator(name, params, desc):
    """Decorator: register a cleaning operator with its param schema."""
    def wrap(fn):
        OPERATORS[name] = {"fn": fn, "params": params, "desc": desc}
        return fn
    return wrap


def _numeric_cols(df, exclude):
    return [c for c in df.columns if c != exclude and pd.api.types.is_numeric_dtype(df[c])]


def _categorical_cols(df, exclude):
    return [c for c in df.columns if c != exclude and not pd.api.types.is_numeric_dtype(df[c])]


@operator(
    "drop_duplicates",
    params={},
    desc="Drop exact duplicate rows.",
)
def drop_duplicates(df, target=None):
    return df.drop_duplicates().reset_index(drop=True)


@operator(
    "drop_missing_rows",
    params={"thresh_pct": "float 0-1: drop a row if its missing fraction exceeds this"},
    desc="Drop rows that are mostly missing.",
)
def drop_missing_rows(df, target=None, thresh_pct=0.5):
    min_non_null = int(np.ceil((1 - thresh_pct) * df.shape[1]))
    return df.dropna(thresh=min_non_null).reset_index(drop=True)


@operator(
    "impute_numeric",
    params={"strategy": "one of: median, mean"},
    desc="Fill missing numeric values.",
)
def impute_numeric(df, target=None, strategy="median"):
    df = df.copy()
    for c in _numeric_cols(df, target):
        if df[c].isna().any():
            fill = df[c].median() if strategy == "median" else df[c].mean()
            df[c] = df[c].fillna(fill)
    return df


@operator(
    "impute_categorical",
    params={"strategy": "one of: mode, constant", "fill_value": "string used when strategy=constant"},
    desc="Fill missing categorical values.",
)
def impute_categorical(df, target=None, strategy="mode", fill_value="missing"):
    df = df.copy()
    for c in _categorical_cols(df, target):
        if df[c].isna().any():
            if strategy == "mode":
                mode = df[c].mode(dropna=True)
                fill = mode.iloc[0] if len(mode) else fill_value
            else:
                fill = fill_value
            df[c] = df[c].fillna(fill)
    return df


@operator(
    "handle_outliers",
    params={"method": "one of: iqr, zscore, none", "action": "one of: clip, drop"},
    desc="Detect and clip/drop numeric outliers.",
)
def handle_outliers(df, target=None, method="iqr", action="clip"):
    if method == "none":
        return df
    df = df.copy()
    for c in _numeric_cols(df, target):
        col = df[c]
        if method == "iqr":
            q1, q3 = col.quantile(0.25), col.quantile(0.75)
            iqr = q3 - q1
            lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        else:  # zscore
            mean, std = col.mean(), col.std()
            if std == 0 or np.isnan(std):
                continue
            lo, hi = mean - 3 * std, mean + 3 * std
        if action == "clip":
            df[c] = col.clip(lo, hi)
        else:  # drop
            df = df[(col >= lo) & (col <= hi)]
    return df.reset_index(drop=True)


@operator(
    "normalize_strings",
    params={"trim": "bool", "case": "one of: lower, upper, none"},
    desc="Trim whitespace and normalize case of string columns.",
)
def normalize_strings(df, target=None, trim=True, case="none"):
    df = df.copy()
    for c in _categorical_cols(df, target):
        if df[c].dtype == object:
            s = df[c].astype("string")
            if trim:
                s = s.str.strip()
            if case == "lower":
                s = s.str.lower()
            elif case == "upper":
                s = s.str.upper()
            df[c] = s
    return df


@operator(
    "coerce_dtypes",
    params={},
    desc="Coerce object columns that are actually numeric/datetime to proper dtypes.",
)
def coerce_dtypes(df, target=None):
    df = df.copy()
    for c in df.columns:
        if c == target or not df[c].dtype == object:
            continue
        coerced = pd.to_numeric(df[c], errors="coerce")
        # only adopt if coercion didn't destroy most of the column
        if coerced.notna().mean() >= max(df[c].notna().mean() - 0.02, 0.0) and coerced.notna().any():
            df[c] = coerced
    return df


@operator(
    "drop_constant_columns",
    params={},
    desc="Drop columns with a single unique value (no signal).",
)
def drop_constant_columns(df, target=None):
    drop = [c for c in df.columns if c != target and df[c].nunique(dropna=False) <= 1]
    return df.drop(columns=drop)


def catalog():
    """Human/LLM-readable operator catalog for the planner prompt."""
    lines = []
    for name, meta in OPERATORS.items():
        params = ", ".join(f"{k} ({v})" for k, v in meta["params"].items()) or "none"
        lines.append(f"- {name}: {meta['desc']} params: {params}")
    return "\n".join(lines)
