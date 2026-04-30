import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from scipy import stats

def evaluate_strategy(df, target, task_type, metric):
    X = pd.get_dummies(df.drop(columns=[target]), drop_first=True)
    y = df[target]
    
    model = RandomForestClassifier() if task_type=="classification" else RandomForestRegressor()
    
    scoring = "f1_weighted" if metric =="f1" else "neg_root_mean_squared_error"
    
    scores = cross_val_score(model, X, y, cv=5, scoring = scoring)
    
    return scores

def compare_strategies(score_dict):
    results = {}
    for name, scores in score_dict.items():
        results[name] = {
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores)),
            "ci95": float(1.96*np.std(scores)/np.sqrt(len(scores)))
            
        }
    return results
        