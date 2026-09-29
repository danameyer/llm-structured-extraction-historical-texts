# LLM Structured Extraction from Historical Texts

This repository contains the code, data, prompts, model configurations, and
evaluation pipeline used for experiments on structured information extraction
from medieval English court records.

The extraction task identifies persons and represents their names, descriptive
attributes, family relations, and legal relationships according to a predefined
JSON schema.

## Setup

The experiments require Python 3.10 or newer.

```bash
git clone <repository-url>
cd llm-structured-extraction-historical-texts

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file in the repository root:

```env
PROJECT_BASE_DIR=/absolute/path/to/llm-structured-extraction-historical-texts

OPENAI_API_KEY=<your-openai-api-key>

OLLAMA_URL=<your-ollama-server-url>
OLLAMA_TOKEN=<your-ollama-token>

# optional; defaults to 600 seconds
OLLAMA_TIMEOUT_SECONDS=600
```

The OpenAI models require `OPENAI_API_KEY`. The open-weight models are accessed
through an Ollama server and require `OLLAMA_URL` and `OLLAMA_TOKEN`. 

The exact model configurations used in the experiments are defined in:

```text
function_calling_setup/models/models.py
```

## Data

The evaluation corpus contains 100 annotated court records taken from 
M. T. Clanchy's 1973 edition *The Roll and Writ File of the Berkshire Eyre of 1248*

The fixed experimental subsets are stored in:

```text
evaluation_results/txt_files_prompt_selection/      # 30 development documents
evaluation_results/txt_files_main_evaluation/       # 70 held-out documents
evaluation_results/txt_files_reasoning_evaluation/  # 50-document subset
evaluation_results/txt_files_repeatability/         # 30-document subset
evaluation_results/ground_truth/                     # ground-truth annotations
```

The development/main split is fixed. The split folders can be regenerated with:

```bash
python data/split_data.py
```

This uses a fixed random seed (`42`). Running the script recreates the split
directories.

The few-shot demonstration is stored in:

```text
test_data/demonstrations/demonstration_1.txt
```

## Experiments

The main experiment runner is:

```text
experiments/experiment_function_calling_evaluation.py
```

The study consists of the following experiments:

| Experiment | Documents | Models / conditions | Purpose |
| --- | ---: | --- | --- |
| 1. Prompt selection | 30 | Luna, Qwen; 3 prompt variants | Select prompting strategy |
| 2A. Structured-output comparison | 30 | 6 models; 3 common output modes + OpenAI strict tool calling | Select output mechanism |
| 2B. Main model comparison | 70 | 6 models | Compare models on held-out data |
| 3. Reasoning comparison | 50 | Luna, Qwen, GPT-OSS, Gemma; baseline vs. reasoning | Evaluate reasoning settings |
| 4. Repeatability | 30 | Luna, Qwen; 3 runs | Assess output variability |

The configuration selected on the development set and used for the subsequent
experiments is defined in `experiments/experiment_config.py`:

```python
MAIN_COMPARISON_PROMPT = "best_prompt_direct_structured_outputs"
```

This corresponds to the selected few-shot prompt with schema-constrained
structured output.

### Running individual experiments

Run commands from the repository root.

Prompt selection:

```bash
python -c "from experiments.experiment_function_calling_evaluation import _run_prompt_selection; _run_prompt_selection()"
```

Structured-output comparison:

```bash
python -c "from experiments.experiment_function_calling_evaluation import _run_structured_output_comparison; _run_structured_output_comparison()"
```

Main model comparison:

```bash
python -c "from experiments.experiment_function_calling_evaluation import _run_main_model_comparison; _run_main_model_comparison()"
```

Reasoning comparison:

```bash
python -c "from experiments.experiment_function_calling_evaluation import _run_reasoning_comparison; _run_reasoning_comparison()"
```

Repeatability experiment:

```bash
python -c "from experiments.experiment_function_calling_evaluation import _run_repeatability; _run_repeatability()"
```

Running `experiments/experiment_function_calling_evaluation.py` directly
currently executes all experiment stages sequentially.

The model lists and reasoning configurations can be found in
`function_calling_setup/models/models.py`.

## Results

Experiment outputs are stored under:

```text
evaluation_results/results/
```

Each model/condition directory contains (where applicable):

```text
predictions/         generated structured predictions
scores_json/         aggregated evaluation results
scores_txt/          human-readable evaluation output
runtime/             runtime information
costs/               token and API-cost information
retry_stats/         generation/retry statistics
logs_failed_files/   failed-document logs
experiment_config.json
```

`experiment_config.json` records the model configuration, prompt condition,
dataset split, and result group used for each experiment.

Predictions are validated against the annotation schema before evaluation.
Failed generations are included in the evaluation as failures.

### Re-running experiments

The repository contains the outputs of the reported experiments. By default,
existing prediction files are not regenerated.

For a completely independent reproduction, first preserve the reported results
and start with an empty result directory, for example:

```bash
mv evaluation_results/results evaluation_results/results_reported
mkdir -p evaluation_results/results
```

Then run the required experiment commands above.

This is preferable to repeatedly restarting a partially completed condition,
because the experiments use a fixed retry limit and completed predictions are
skipped when a run is resumed.

## Repository structure

```text
data/                         corpus data and split-generation script
evaluation/                   matching, validation, and evaluation
evaluation_results/           fixed subsets, ground truth, and results
experiments/                  experiment definitions and runner
function_calling_setup/       providers, model configuration, retry/runtime tracking
function_definition/          extraction function and output schema
prompting/                    prompt construction
test_data/                    few-shot demonstration and test resources
visualisation/                scripts for generating result figures
```

## License

See [LICENSE](LICENSE).