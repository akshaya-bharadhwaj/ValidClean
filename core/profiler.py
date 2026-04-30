# Agent 1: Data Profiler

import pandas as pd

def profile_dataset(df):
    profile = {
        "num_rows": len(df),
        "num_columns": df.shape[1],
        "duplicate_row_pct": float(df.duplicated().mean()),
        "columns":{}
    }
    
    for col in df.columns:
        s = df[col]
        col_info = {
            "dtype": str(s.dtype),
            "missing_pct": float(s.isna().mean()),
            "n_unique": int(s.nunique()),
            "sample_values": s.dropna().astype(str).unique()[:5].tolist()
        }
        
        if pd.api.types.is_numeric_dtype(s):
            col_info.update({
                "mean": float(s.mean()),
                "std": float(s.std()),
                "min": float(s.min()),
                "max": float(s.max())
            })

        profile["columns"][col] = col_info
        
    return profile


