import numpy as np
import pandas as pd

def apply_cleaning_code(df, code):
    local_vars = {"df": df.copy(), "pd": pd, "np": np}  # add pd and np here
    try:
        exec(code, {}, local_vars)
        return local_vars["df"], None
    except Exception as e:
        return None, str(e)