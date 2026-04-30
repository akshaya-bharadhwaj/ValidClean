# Agent 3: Strategy Generator
import json
from core.llm_client import call_llm

STRATEGY_TYPES = [
    ("Minimal", "Only drop rows with missing values. No other changes."),
    ("Conservative", "Impute missing values with median/mode. Drop duplicates."),
    ("Aggressive", "Impute, drop outliers, encode categoricals, scale numerics."),
    ("Task-Optimized", "Focus only on columns relevant to the target variable."),
    ("Ensemble-style", "Combine imputation, encoding, and feature selection."),
]

def generate_single_strategy(name, description, profile, task_spec):
    prompt = f"""
    You are a data scientist. Write pandas/numpy code to clean a dataset.
    
    Strategy: {name} - {description}
    
    Target column: {task_spec["target_column"]}
    Columns: {list(profile["columns"].keys())}
    
    Rules:
    - Use only pandas/numpy
    - Modify df directly
    - No file loading, no function definitions
    
    Return ONLY this JSON, no explanation:
    {{"name": "{name}", "description": "{description}", "code": "..."}}
    """
    for attempt in range(3):  # retry up to 3 times
        try:
            response = call_llm(prompt, temperature=0.7)
            return json.loads(response)
        except json.JSONDecodeError:
            print(f"  Retrying {name} (attempt {attempt+2})...")
    raise ValueError(f"Failed to generate valid JSON for {name} after 3 attempts")

def generate_strategies(profile, task_spec, n_strategies=5):
    strategies = []
    for name, description in STRATEGY_TYPES:
        print(f"  Generating {name} strategy...")
        try:
            strat = generate_single_strategy(name, description, profile, task_spec)
            strategies.append(strat)
        except Exception as e:
            print(f"  Skipping {name}: {e}")
    return strategies