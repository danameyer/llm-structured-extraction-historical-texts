import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List
from evaluation.json_comparison import JsonComparison
from experiments.function_calling_all_principles_few_shot import ExperimentAllPrinciplesFewShotPrompt
from experiments.function_calling_all_principles_zero_shot import ExperimentAllPrinciplesZeroShotPrompt
from experiments.function_calling_no_principles_base_prompt import ExperimentFunctionCallingNoPrinciplesBasePrompt
from experiments.function_calling_optimised_no_function_calling import ExperimentOptimisedPromptNoFunctionCalling
from experiments.function_calling_with_optimised_prompt import ExperimentFunctionCallingWithOptimisedPrompt
from function_calling_setup.chat_file_writer import ChatFileWriter
from function_calling_setup.models.model_config import ModelConfig
from dotenv import load_dotenv
from function_calling_setup.models.models import (
    MAIN_COMPARISON_MODELS,
    OPENAI_MODELS,
    OPENAI_SMOKE_TEST,
    PROMPT_SELECTION_MODELS,
    REASONING_MODEL_PAIRS,
    REPEATABILITY_MODELS, QWEN_36_35B, QWEN_36_35B_REASONING_BASELINE, GPT_OSS_20B_REASONING_BASELINE,
    GEMMA4_26B_REASONING_BASELINE, MISTRAL_SMALL_32_24B, QWEN_36_35B_REASONING_ENABLED, GPT_OSS_20B_REASONING_ENABLED,
    GEMMA4_26B_REASONING_ENABLED
)
from function_calling_setup.providers.provider_factory import create_provider
from experiments.experiment_config import (
    REASONING_BASELINE_RESULT_GROUP,
    REASONING_ENABLED_RESULT_GROUP,
    REASONING_SAMPLE_FOLDER,
    PROMPT_SELECTION_PROMPTS,
    COMMON_STRUCTURED_OUTPUT_PROMPTS,
    OPENAI_ONLY_STRUCTURED_OUTPUT_PROMPTS,
    MAIN_COMPARISON_PROMPT,
    _save_experiment_metadata, MAIN_COMPARISON_RESULT_GROUP, REPEATABILITY_RESULT_GROUPS
)

load_dotenv()

class ExperimentFunctionCallingEvaluation:

    def __init__(
            self,
            prompt_experiment_name: str,
            experiment_dir,
            model_config: ModelConfig,
            regenerate_predictions=False,
    ):
        self.experiment_dir = experiment_dir
        self.prompt_experiment_name = prompt_experiment_name
        self.model_config = model_config
        self.model_name = model_config.model
        self.regenerate_predictions = regenerate_predictions
        self.failed_files = []

    def init_prompt_experiment(
            self,
            demonstrations_files,
            test_files,
            pred_file_name,
    ):
        strict = (self.prompt_experiment_name == "best_prompt_structured_outputs_enabled")
        provider = create_provider(config=self.model_config, strict=strict)

        if self.prompt_experiment_name == "chain_of_thought":
            return ExperimentFunctionCallingWithOptimisedPrompt(
                test_files,
                demonstrations_files,
                provider,
                self.experiment_dir,
                pred_file_name,
            )

        elif self.prompt_experiment_name == "all_principles_zero_shot":
            return ExperimentAllPrinciplesZeroShotPrompt(
                test_files,
                demonstrations_files,
                provider,
                self.experiment_dir,
                pred_file_name,
            )

        elif self.prompt_experiment_name == "all_principles_few_shot":
            return ExperimentAllPrinciplesFewShotPrompt(
                test_files,
                demonstrations_files,
                provider,
                self.experiment_dir,
                pred_file_name,
            )

        elif self.prompt_experiment_name == "no_principles_base_prompt":
            return ExperimentFunctionCallingNoPrinciplesBasePrompt(
                test_files,
                demonstrations_files,
                provider,
                self.experiment_dir,
                pred_file_name,
            )

        elif self.prompt_experiment_name == "best_prompt":
            return ExperimentFunctionCallingWithOptimisedPrompt(
                test_files,
                demonstrations_files,
                provider,
                self.experiment_dir,
                pred_file_name,
            )

        elif self.prompt_experiment_name == "best_prompt_prompt_only_json":
            return ExperimentOptimisedPromptNoFunctionCalling(
                test_files,
                demonstrations_files,
                provider,
                self.experiment_dir,
                pred_file_name,
                final_response_mode="prompted_json",
            )

        elif self.prompt_experiment_name == "best_prompt_direct_structured_outputs":
            return ExperimentOptimisedPromptNoFunctionCalling(
                test_files,
                demonstrations_files,
                provider,
                self.experiment_dir,
                pred_file_name,
                final_response_mode="json_schema",
            )

        elif self.prompt_experiment_name == "best_prompt_structured_outputs_disabled":
            return ExperimentFunctionCallingWithOptimisedPrompt(
                test_files,
                demonstrations_files,
                provider,
                self.experiment_dir,
                pred_file_name,
            )

        elif self.prompt_experiment_name == "best_prompt_structured_outputs_enabled":
            return ExperimentFunctionCallingWithOptimisedPrompt(
                test_files,
                demonstrations_files,
                provider,
                self.experiment_dir,
                pred_file_name,
            )

        else:
            raise ValueError("Wrong prompt name: " + self.prompt_experiment_name)

    @staticmethod
    def get_base_directory():
        base_dir = os.getenv("PROJECT_BASE_DIR")

        if not base_dir:
            raise ValueError("PROJECT_BASE_DIR is not set.")

        return Path(base_dir)

    @staticmethod
    def get_ground_truth_folder(base_dir):
        return os.path.join(base_dir, "evaluation_results", "ground_truth")

    @staticmethod
    def get_demonstrations_folder(base_dir):
        return os.path.join(base_dir, "test_data", "demonstrations")

    @staticmethod
    def get_sample_folder(base_dir, sample_folder_name):
        return os.path.join(base_dir, "evaluation_results", sample_folder_name)

    @staticmethod
    def create_predictions_folder(base_dir):
        predictions_folder = os.path.join(base_dir, "predictions")
        os.makedirs(predictions_folder, exist_ok=True)
        return predictions_folder

    @staticmethod
    def create_scores_txt_folder(base_dir):
        output_folder = os.path.join(base_dir, "scores_txt")
        os.makedirs(output_folder, exist_ok=True)
        return output_folder

    @staticmethod
    def create_scores_json_folder(base_dir):
        output_folder = os.path.join(base_dir, "scores_json")
        os.makedirs(output_folder, exist_ok=True)
        return output_folder

    @staticmethod
    def create_costs_folder(experiment_dir):
        costs_folder = os.path.join(experiment_dir, "costs")
        os.makedirs(costs_folder, exist_ok=True)
        return costs_folder

    @staticmethod
    def create_runtime_folder(experiment_dir):
        runtime_folder = os.path.join(experiment_dir, "runtime")
        os.makedirs(runtime_folder, exist_ok=True)
        return runtime_folder

    @staticmethod
    def create_logs_folder(experiment_dir):
        logs_folder = os.path.join(experiment_dir, "logs_failed_files")
        os.makedirs(logs_folder, exist_ok=True)
        return logs_folder

    @staticmethod
    def create_retry_stats_folder(experiment_dir):
        retry_stats_folder = os.path.join(experiment_dir, "retry_stats")
        os.makedirs(retry_stats_folder, exist_ok=True)
        return retry_stats_folder

    def log_failed_files(self, experiment_dir):
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        log_filename = f'failed_files_{timestamp}.txt'
        logs_directory = self.create_logs_folder(experiment_dir)
        log_file_path = os.path.join(logs_directory, log_filename)

        with open(log_file_path, 'w') as log_file:
            log_file.write(f"Failed files count: {len(self.failed_files)}\n\n")
            for failed_file in self.failed_files:
                log_file.write(f"{failed_file}\n")

    def aggregate_retry_stats(self, experiment_dir, expected_base_names):
        retry_directory = self.create_retry_stats_folder(experiment_dir)
        retry_summaries = []

        for base_name in expected_base_names:
            retry_file = os.path.join(retry_directory, f"retry_summary_pred_{base_name}.json")

            if not os.path.exists(retry_file):
                continue

            with open(retry_file, "r", encoding="utf-8") as file:
                retry_summaries.append(json.load(file))

        if not retry_summaries:
            return None

        attempts = [summary["attempts"] for summary in retry_summaries]
        successful_summaries = [summary for summary in retry_summaries if summary["success"]]
        first_attempt_successes = sum(1 for summary in retry_summaries if summary["success"] and summary["attempts"] == 1)
        total_retries = sum(summary["retries"] for summary in retry_summaries)

        return {
            "documents": len(retry_summaries),
            "total_attempts": sum(attempts),
            "total_retries": total_retries,
            "first_attempt_successes": first_attempt_successes,
            "first_attempt_success_rate": first_attempt_successes / len(retry_summaries),
            "eventual_successes": len(successful_summaries),
            "failed_documents": len(retry_summaries) - len(successful_summaries),
            "mean_attempts": sum(attempts) / len(attempts),
            "mean_attempts_successful": (
                sum(summary["attempts"] for summary in successful_summaries) / len(successful_summaries)
                if successful_summaries
                else 0.0
            ),
            "max_attempts_used": max(attempts),
        }

    def process_files(
            self,
            text_files_folder,
            demonstrations_folder,
            predictions_folder,
            max_files=None,
    ):
        file_names = sorted(os.listdir(text_files_folder))

        if max_files is not None:
            file_names = file_names[:max_files]

        expected_base_names = []

        for file_name in file_names:
            path_to_text_file = os.path.join(text_files_folder, file_name)

            if os.path.isfile(path_to_text_file):
                base_name = os.path.splitext(file_name)[0]
                expected_base_names.append(base_name)

                try:
                    self.process_single_file(
                        path_to_text_file,
                        demonstrations_folder,
                        predictions_folder
                    )

                except Exception as e:
                    print(f"Could not complete file {path_to_text_file} because of {type(e).__name__}: {e}. Will continue with next file.", file=sys.stderr)

                    self.failed_files.append(path_to_text_file)

        if self.failed_files:
            self.log_failed_files(self.experiment_dir)

        return expected_base_names

    def process_single_file(self, path_to_text_file, demonstrations_folder, predictions_folder):
        base_name = os.path.splitext(os.path.basename(path_to_text_file))[0]
        pred_filename = f'pred_{base_name}.json'
        output_path_response = os.path.join(predictions_folder, pred_filename)
        base_file_name = os.path.splitext(os.path.basename(pred_filename))[0]

        if os.path.exists(output_path_response):
            if self.regenerate_predictions:
                os.remove(output_path_response)
                print(f"Removed old prediction for {base_name} before regeneration.")
            else:
                print(f"Prediction for {base_name} already exists. Skipping regeneration.")
                return

        test_files = [path_to_text_file]
        demonstrations = [os.path.join(demonstrations_folder, "demonstration_1.txt")]
        experiment_function_calling = (
            self.init_prompt_experiment(
                demonstrations,
                test_files,
                pred_file_name=base_name,
            )
        )

        try:
            response_json = experiment_function_calling.run()
            response_json_str = json.dumps(response_json, indent=4)
            chat_file_writer = ChatFileWriter(self.experiment_dir)

            chat_file_writer.save_response(
                response_json_str,
                output_path_response,
                timestamp=False,
                append=False,
            )

            print(
                f"Processed {os.path.basename(path_to_text_file)}: Saved predictions to {pred_filename}"
            )

        finally:
            dialogue = experiment_function_calling.dialogue
            cost_summary = dialogue.token_counter.get_cost_summary(pred_filename)
            costs_directory = self.create_costs_folder(self.experiment_dir)

            self.save_costs_to_file(
                cost_summary,
                costs_directory,
                f"cost_summary_{base_file_name}.json",
            )

            runtime_summary = dialogue.runtime_calculator.get_runtime_summary(pred_filename)
            runtime_directory = self.create_runtime_folder(self.experiment_dir)

            self.save_runtime_to_file(
                runtime_summary,
                runtime_directory,
                f"runtime_summary_{base_file_name}.json",
            )

            retry_summary = dialogue.retry_tracker.get_summary(pred_filename)
            retry_directory = self.create_retry_stats_folder(self.experiment_dir)

            self.save_retry_summary(
                retry_summary,
                retry_directory,
                f"retry_summary_{base_file_name}.json",
            )

    @staticmethod
    def save_costs_to_file(costs, directory: str, filename):
        costs_as_string = json.dumps(costs)
        file_path = os.path.join(directory, filename)
        with open(file_path, 'w') as f:
            f.write(costs_as_string)

    @staticmethod
    def save_runtime_to_file(runtime, directory: str, filename_runtime):
        runtime_as_string = json.dumps(runtime)
        file_path = os.path.join(directory, filename_runtime)
        with open(file_path, 'w') as f:
            f.write(runtime_as_string)

    @staticmethod
    def save_retry_summary(retry_summary, directory, filename):
        file_path = os.path.join(directory, filename)

        with open(file_path, "w", encoding="utf-8") as file:
            file.write(json.dumps(retry_summary, indent=4))

    def run(self, exclusions: List[List[str]], sample_folder_name, max_files=None):
        base_dir = self.get_base_directory()
        sample_folder = self.get_sample_folder(base_dir, sample_folder_name)
        ground_truth_folder = self.get_ground_truth_folder(base_dir)
        demonstrations_folder = self.get_demonstrations_folder(base_dir)
        predictions_folder = self.create_predictions_folder(self.experiment_dir)
        scores_txt_folder = self.create_scores_txt_folder(self.experiment_dir)
        scores_json_folder = self.create_scores_json_folder(self.experiment_dir)
        expected_base_names = self.process_files(
            sample_folder,
            demonstrations_folder,
            predictions_folder,
            max_files=max_files,
        )

        retry_summary = self.aggregate_retry_stats(self.experiment_dir, expected_base_names)
        json_comparison = JsonComparison()
        json_comparison.perform_json_comparison(
            ground_truth_folder,
            predictions_folder,
            scores_txt_folder,
            scores_json_folder,
            exclusions,
            expected_base_names=expected_base_names,
            retry_summary=retry_summary
        )


def _prepare_and_run_experiment(
        model_config: ModelConfig,
        prompt_name: str,
        sample_folder_name: str,
        max_files=None,
        result_group_name: str | None = None,
):
    regenerate_predictions = False

    exclusions = [
        ["root['id']"],
        ["root['id']", "root['cognomen']"],
        ["root['id']", "root['legal_relationship']"],
        ["root['id']", "root['place_of_origin']"],
        ["root['id']", "root['family_relations']"],
        ["root['id']", "root['title']"],
        ["root['id']", "root['profession']"],
    ]

    base_dir = ExperimentFunctionCallingEvaluation.get_base_directory()
    result_group = result_group_name if result_group_name is not None else prompt_name
    experiment_folder =  base_dir / "evaluation_results" / "results" / result_group / model_config.result_dir
    experiment_folder.mkdir(parents=True, exist_ok=True)

    _save_experiment_metadata(
        experiment_folder=experiment_folder,
        model_config=model_config,
        prompt_name=prompt_name,
        sample_folder_name=sample_folder_name,
        result_group_name=result_group
    )

    experiment = ExperimentFunctionCallingEvaluation(
        prompt_experiment_name=prompt_name,
        model_config=model_config,
        regenerate_predictions=regenerate_predictions,
        experiment_dir=experiment_folder
    )

    experiment.run(
        exclusions=exclusions,
        sample_folder_name=sample_folder_name,
        max_files=max_files
    )


def _run_models_for_prompt(
        models: list[ModelConfig],
        prompt_name: str,
        sample_folder_name: str,
        result_group_name: str | None = None
):
    for model_config in models:
        _prepare_and_run_experiment(
            model_config=model_config,
            prompt_name=prompt_name,
            sample_folder_name=sample_folder_name,
            result_group_name=result_group_name,
        )


def _run_prompt_selection():
    sample_folder_name = "txt_files_prompt_selection"

    for prompt_name in PROMPT_SELECTION_PROMPTS:
        _run_models_for_prompt(
            models=PROMPT_SELECTION_MODELS,
            prompt_name=prompt_name,
            sample_folder_name=sample_folder_name,
        )


def _run_structured_output_comparison():
    sample_folder_name = "txt_files_prompt_selection"

    for prompt_name in COMMON_STRUCTURED_OUTPUT_PROMPTS:
        _run_models_for_prompt(
            models=MAIN_COMPARISON_MODELS,
            prompt_name=prompt_name,
            sample_folder_name=sample_folder_name
        )

    for prompt_name in OPENAI_ONLY_STRUCTURED_OUTPUT_PROMPTS:
        _run_models_for_prompt(
            models=OPENAI_MODELS,
            prompt_name=prompt_name,
            sample_folder_name=sample_folder_name
        )


def _run_main_model_comparison():
    _run_models_for_prompt(
        models=MAIN_COMPARISON_MODELS,
        prompt_name=MAIN_COMPARISON_PROMPT,
        sample_folder_name="txt_files_main_evaluation",
        result_group_name=MAIN_COMPARISON_RESULT_GROUP,
    )

def _run_reasoning_comparison():
    for (baseline_config, reasoning_config) in REASONING_MODEL_PAIRS:

        _prepare_and_run_experiment(
            model_config=baseline_config,
            prompt_name=MAIN_COMPARISON_PROMPT,
            sample_folder_name=REASONING_SAMPLE_FOLDER,
            result_group_name=REASONING_BASELINE_RESULT_GROUP,
        )

        _prepare_and_run_experiment(
            model_config=reasoning_config,
            prompt_name=MAIN_COMPARISON_PROMPT,
            sample_folder_name=REASONING_SAMPLE_FOLDER,
            result_group_name=REASONING_ENABLED_RESULT_GROUP,
        )


def _run_repeatability():
    sample_folder_name = "txt_files_repeatability"

    for result_group_name in REPEATABILITY_RESULT_GROUPS:
        _run_models_for_prompt(
            models=REPEATABILITY_MODELS,
            prompt_name=MAIN_COMPARISON_PROMPT,
            sample_folder_name=sample_folder_name,
            result_group_name=result_group_name,
        )


def _openai_smoke_test():
    prompt_names = [
        "best_prompt_prompt_only_json",
        "best_prompt_direct_structured_outputs",
        "best_prompt_structured_outputs_disabled",
        "best_prompt_structured_outputs_enabled",
    ]

    for model_config in OPENAI_MODELS:
        for prompt_name in prompt_names:
            _prepare_and_run_experiment(
                model_config=model_config,
                prompt_name=prompt_name,
                sample_folder_name="txt_files_function_calling_evaluation",
                max_files=2,
                result_group_name=f"openai_smoke_{prompt_name}"
            )

def _ollama_smoke_test():
    baseline_models = [
        QWEN_36_35B_REASONING_BASELINE,
        GPT_OSS_20B_REASONING_BASELINE,
        GEMMA4_26B_REASONING_BASELINE,
        MISTRAL_SMALL_32_24B
    ]

    reasoning_models = [
        QWEN_36_35B_REASONING_ENABLED,
        GPT_OSS_20B_REASONING_ENABLED,
        GEMMA4_26B_REASONING_ENABLED
    ]

    for model_config in baseline_models:
        for prompt_name in COMMON_STRUCTURED_OUTPUT_PROMPTS:
            _prepare_and_run_experiment(
                model_config=model_config,
                prompt_name=prompt_name,
                sample_folder_name="txt_files_function_calling_evaluation",
                max_files=1,
                result_group_name=f"ollama_smoke_baseline_{prompt_name}"
            )

    for model_config in reasoning_models:
        for prompt_name in COMMON_STRUCTURED_OUTPUT_PROMPTS:
            _prepare_and_run_experiment(
                model_config=model_config,
                prompt_name=prompt_name,
                sample_folder_name="txt_files_function_calling_evaluation",
                max_files=1,
                result_group_name=f"ollama_smoke_reasoning_{prompt_name}"
            )

def _mistral_tool_smoke_test():
    _prepare_and_run_experiment(
        model_config=MISTRAL_SMALL_32_24B,
        prompt_name="best_prompt_structured_outputs_disabled",
        sample_folder_name="txt_files_function_calling_evaluation",
        max_files=1,
        result_group_name="ollama_smoke_mistral_tool_debug"
    )

def _mistral_other_formats_smoke_test():
    prompt_names = [
        "best_prompt_prompt_only_json",
        "best_prompt_direct_structured_outputs",
    ]

    for prompt_name in prompt_names:
        _prepare_and_run_experiment(
            model_config=MISTRAL_SMALL_32_24B,
            prompt_name=prompt_name,
            sample_folder_name="txt_files_function_calling_evaluation",
            max_files=1,
            result_group_name=f"ollama_smoke_mistral_{prompt_name}"
        )


if __name__ == "__main__":
    # _openai_smoke_test()
    # _ollama_smoke_test()
    # _mistral_tool_smoke_test()
    _mistral_other_formats_smoke_test()

    # _run_prompt_selection()
    # _run_structured_output_comparison()
    # _run_main_model_comparison()
    # _run_reasoning_comparison()
    # _run_repeatability()
