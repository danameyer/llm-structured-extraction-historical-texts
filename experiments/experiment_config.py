import json
from pathlib import Path
from function_calling_setup.models.model_config import ModelConfig

PROMPT_SELECTION_PROMPTS = [
    "all_principles_zero_shot",
    "all_principles_few_shot",
    "no_principles_base_prompt"
]
COMMON_STRUCTURED_OUTPUT_PROMPTS = [
    "best_prompt_prompt_only_json",
    "best_prompt_direct_structured_outputs",
    "best_prompt_structured_outputs_disabled"
]
OPENAI_ONLY_STRUCTURED_OUTPUT_PROMPTS = ["best_prompt_structured_outputs_enabled"]
MAIN_COMPARISON_PROMPT = "best_prompt_direct_structured_outputs"
REASONING_BASELINE_RESULT_GROUP = "reasoning_baseline"
REASONING_ENABLED_RESULT_GROUP = "reasoning_enabled"
REASONING_SAMPLE_FOLDER = "txt_files_reasoning_evaluation"
MAIN_COMPARISON_RESULT_GROUP = "main_model_comparison"
REPEATABILITY_RESULT_GROUPS = ["repeatability_run_1", "repeatability_run_2", "repeatability_run_3"]

def _save_experiment_metadata(
        experiment_folder: Path,
        model_config: ModelConfig,
        prompt_name: str,
        sample_folder_name: str,
        result_group_name: str | None = None,
) -> None:
    metadata = {
        "model_config": {
            "provider": model_config.provider,
            "model": model_config.model,
            "result_name": model_config.result_name,
            "reasoning_effort": model_config.reasoning_effort,
            "think": model_config.think,
        },
        "prompt_name": prompt_name,
        "sample_folder_name": sample_folder_name,
        "result_group_name": result_group_name or prompt_name,
    }

    metadata_path = experiment_folder / "experiment_config.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=4), encoding="utf-8")
