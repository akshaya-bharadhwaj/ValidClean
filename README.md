# ValidClean 
> A multi-agent LLM system for scientifically validated, automated data cleaning

ValidClean eliminates the guesswork from data preprocessing. Instead of applying a 
one-size-fits-all cleaning pipeline, it generates multiple strategies and uses 
statistical A/B testing to determine which one actually improves your ML model.

---

## How It Works

ValidClean uses a 5-agent pipeline:

| Agent | Role |
|---|---|
| Profiler | Analyzes dataset structure, missing values, and statistics |
| Task Interpreter | Infers task type (classification/regression), target column, and metric |
| Strategy Generator | Generates 5 diverse cleaning strategies via LLM |
| Executor | Safely runs each strategy in an isolated environment |
| Explainer | Explains why the winning strategy outperformed the others |

---

## Results (Titanic Dataset)

| Strategy | Mean F1 | Std |
|---|---|---|
| MinimalCleaning | 0.819 | 0.027 |
| ConservativeCleaning | TBD | TBD |
| AggressiveCleaning | TBD | TBD |
| Task-Optimized | TBD | TBD |

---

## Quickstart

### Prerequisites
- Python 3.12+
- [LM Studio](https://lmstudio.ai/) running locally with `phi-3-mini-4k-instruct` loaded

### Installation

git clone https://github.com/YOUR_USERNAME/ValidClean.git
cd ValidClean
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

### Run

1. Start LM Studio local server on `http://127.0.0.1:1234`
2. Load `phi-3-mini-4k-instruct` in LM Studio
3. Add your dataset to `data/raw/`
4. Update `DATA_PATH` and `TASK_DESCRIPTION` in `experiments/run_validclean.py`
5. Run:

python -m experiments.run_validclean

---

## Project Structure

ValidClean/
├── core/
│   ├── profiler.py          # Agent 1: Dataset profiling
│   ├── task_interpreter.py  # Agent 2: Task inference
│   ├── strategy_agent.py    # Agent 3: Strategy generation
│   ├── executor.py          # Agent 4: Safe code execution
│   ├── explainer.py         # Agent 5: Result explanation
│   └── llm_client.py        # LM Studio API client
├── evaluation/
│   └── ab_testing.py        # K-fold CV + statistical comparison
├── experiments/
│   └── run_validclean.py    # Main entry point
├── data/raw/                # Your datasets go here (not tracked by git)
└── README.md

---

## Roadmap
- [ ] Add paired t-tests and Cohen's d for statistical significance
- [ ] Support multiple datasets beyond Titanic
- [ ] Switch LLM backend to stronger model (gemma-2-9b)
- [ ] Add CLI interface for any CSV + task description
- [ ] Grad Slam demo mode

---

## Built With
Python · pandas · scikit-learn · scipy · LM Studio · phi-3-mini-4k-instruct
