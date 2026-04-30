# Agnent 5: Explanation Agent

from core.llm_client import call_llm
import json

def generate_explanation(results_table, winning_strategy):
    prompt = f"""
    We tested multiple data cleaning strategies.
    
    Results:
    {json.dumps(results_table, indent=2)}
    
    Winning Strategy:
    {json.dumps(winning_strategy, indent=2)}
    
    Explain in simple terms:
    - Why this strategy likely worked the best
    - What trade-offs it made
    - Why others underperformed
    """
    
    return call_llm(prompt, temperature=0.3)

