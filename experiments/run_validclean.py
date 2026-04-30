import pandas as pd
from core.profiler import profile_dataset
from core.task_interpreter import interpret_task
from core.strategy_agent import generate_strategies
from core.executor import apply_cleaning_code
from evaluation.ab_testing import evaluate_strategy, compare_strategies
from core.explainer import generate_explanation



DATA_PATH  = "data/raw/Titanic.csv"
TASK_DESCRIPTION = "Predict survival of passengers"

def main():
    print("Loading Dataset")
    df = pd.read_csv(DATA_PATH)
    
    print("Profiling Dataset")
    profile = profile_dataset(df)
    
    print("Interpreting task")
    task_spec = interpret_task(TASK_DESCRIPTION, profile)
    
    print("Generating strategies")
    strategies = generate_strategies(profile, task_spec)
    
    scores = {}
    strategy_objects = {}
    
    print("Running A/B tests")
    for strat in strategies:
        print(f"Testing {strat['name']}")
        cleaned_df, err = apply_cleaning_code(df, strat["code"])
        if err:
            print("Failed:", err)
            continue
        # make sure target column survived cleaning
        if task_spec["target_column"] not in cleaned_df.columns:
            print(f"Skipping {strat['name']}: target column was dropped by cleaning code")
            continue
        s = evaluate_strategy(cleaned_df, task_spec["target_column"], task_spec["task_type"], task_spec["metric"])
        scores[strat["name"]] = s
        strategy_objects[strat["name"]] = strat
        
    results_table = compare_strategies(scores)
    
    best_name = max(results_table, key = lambda x: results_table[x]["mean"])
    explanation = generate_explanation(results_table, strategy_objects[best_name])
    
    
    print("Best Strategy:", best_name)
    print(results_table[best_name])
    print("Explanation:", explanation)
    
if __name__ == "__main__":
    main()