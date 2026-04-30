# Agent 2: Task Interpreter (LLM)

import json
from core.llm_client import call_llm
import re

def extract_json(text):
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in LLM response:\n{text}")
    return json.loads(match.group())

def interpret_task(task_description, profile):
    prompt = f"""
    You are an ML expert.
    User task description:
    "{task_description}"
    Dataset olumns:
    {list(profile["columns"].keys())}
    
    Infer:
    1. Task type (classification or regression)
    2. Target column name
    3. Best evaluation metric
    
    Return JSON:
    {{
        "task_type": "...",
        "target_column": "...",
        "metric": "f1 or rmse"
    }}
    Return ONLY valid minified JSON.
    No prose.
    No explanation.
    No markdown.
    No backticks.
    
    """
    
    response = call_llm(prompt, temperature=0.2)
    return extract_json(response)
