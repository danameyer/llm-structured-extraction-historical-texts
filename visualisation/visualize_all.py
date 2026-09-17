import os
from pathlib import Path
from dotenv import load_dotenv
from visualisation.box_plot_fuzzy_score import BoxPlotFuzzyScore
from visualisation.general_structured_output_retrieval import GeneralStructuredOutputRetrieval
from visualisation.plot_accuracy_document_size import ScatterPlot
from visualisation.plot_costs import CostPlot
from visualisation.plot_runtime import RuntimePlot
from visualisation.plot_with_error_bars import PlotWithErrorBars
from visualisation.scores_error_types import ErrorTypePlot
from visualisation.scores_for_exclude_paths import ExcludePathsPlot
from visualisation.scores_overall_accuracy import OverallAccuracy
from visualisation.plot_reasoning import ReasoningPlot
from function_calling_setup.models import (
    MAIN_COMPARISON_MODELS,
    OPENAI_MODELS,
    PROMPT_SELECTION_MODELS,
    REASONING_MODEL_PAIRS,
    REPEATABILITY_MODELS
)
from experiments.experiment_config import (
    COMMON_STRUCTURED_OUTPUT_PROMPTS,
    OPENAI_ONLY_STRUCTURED_OUTPUT_PROMPTS,
    PROMPT_SELECTION_PROMPTS,
    REASONING_BASELINE_RESULT_GROUP,
    REASONING_ENABLED_RESULT_GROUP, MAIN_COMPARISON_RESULT_GROUP, REPEATABILITY_RESULT_GROUPS,
)

BASELINE_EXCLUSION = ["root['id']"]

def get_model_dirs(models):
    return [model.result_dir for model in models]

def get_directories():
    load_dotenv()
    base_dir = os.getenv('PROJECT_BASE_DIR')

    if not base_dir:
        raise ValueError("PROJECT_BASE_DIR is not set.")

    base_dir = Path(base_dir)
    results_directory = base_dir / 'evaluation_results' / 'results'
    text_directory = base_dir / 'evaluation_results' / 'txt_files_function_calling_evaluation'

    return results_directory, text_directory

def get_available_model_dirs(results_directory, prompt_name, models):
    return [
        model.result_dir for model in models
        if (results_directory / prompt_name / model.result_dir).exists()
    ]

def get_available_model_dirs_for_prompts(results_directory, prompt_names, models):
    return [
        model.result_dir for model in models
        if any((results_directory / prompt_name / model.result_dir).exists()
            for prompt_name in prompt_names
        )
    ]

def plot_prompt_selection_accuracy(results_directory):
    overall_accuracy = OverallAccuracy()

    model_dirs = get_available_model_dirs_for_prompts(
        results_directory,
        PROMPT_SELECTION_PROMPTS,
        PROMPT_SELECTION_MODELS,
    )

    if not model_dirs:
        return

    scores_by_prompt = {}

    for prompt_name in PROMPT_SELECTION_PROMPTS:
        models_directory = results_directory / prompt_name

        available_model_dirs = [
            model_dir for model_dir in model_dirs
            if (models_directory / model_dir).exists()
        ]

        if not available_model_dirs:
            continue

        (model_names, exact_scores, fuzzy_scores) = overall_accuracy.extract_overall_accuracy(
            models_directory,
            available_model_dirs
        )

        scores_by_prompt[prompt_name] = {
            model_name: {"exact": exact_score, "fuzzy": fuzzy_score}
            for (model_name, exact_score, fuzzy_score) in zip(model_names, exact_scores, fuzzy_scores)
        }

    if not scores_by_prompt:
        return

    overall_accuracy.plot_prompt_selection_comparison(scores_by_prompt, model_dirs)

def plot_main_model_accuracy(results_directory):
    overall_accuracy = OverallAccuracy()
    models_directory = results_directory / MAIN_COMPARISON_RESULT_GROUP
    model_dirs = get_available_model_dirs(
        results_directory,
        MAIN_COMPARISON_RESULT_GROUP,
        MAIN_COMPARISON_MODELS
    )

    model_names, exact_scores, fuzzy_scores = (
        overall_accuracy.extract_overall_accuracy(models_directory, model_dirs)
    )

    overall_accuracy.plot_overall_accuracy(
        model_names,
        exact_scores,
        fuzzy_scores,
        "main_model_comparison",
    )

def plot_main_model_costs(results_directory):
    cost_plot = CostPlot()
    models_directory = results_directory / MAIN_COMPARISON_RESULT_GROUP

    model_dirs = get_available_model_dirs(
        results_directory,
        MAIN_COMPARISON_RESULT_GROUP,
        OPENAI_MODELS,
    )

    if not model_dirs:
        return

    model_names, costs = cost_plot.calculate_costs(models_directory, model_dirs)
    cost_plot.plot_costs(model_names, costs,"main_model_comparison")


def plot_prompt_selection_runtime(results_directory):
    runtime_plot = RuntimePlot()
    prompt_names = PROMPT_SELECTION_PROMPTS

    for prompt_name in prompt_names:
        models_directory = results_directory / prompt_name
        model_dirs = get_available_model_dirs(
            results_directory,
            prompt_name,
            PROMPT_SELECTION_MODELS
        )
        model_names, runtimes = runtime_plot.calculate_runtime(models_directory, model_dirs)
        runtime_plot.plot_runtime(model_names, runtimes, prompt_name)

def plot_main_model_runtime(results_directory):
    runtime_plot = RuntimePlot()
    models_directory = results_directory / MAIN_COMPARISON_RESULT_GROUP

    model_dirs = get_available_model_dirs(
        results_directory,
        MAIN_COMPARISON_RESULT_GROUP,
        MAIN_COMPARISON_MODELS
    )

    if not model_dirs:
        return

    model_names, runtimes = runtime_plot.calculate_runtime(models_directory, model_dirs)
    runtime_plot.plot_runtime(model_names, runtimes,"main_model_comparison")


def plot_repeated_run_accuracy(results_directory):
    error_bar_plot = PlotWithErrorBars()

    model_dirs = get_available_model_dirs_for_prompts(
        results_directory,
        REPEATABILITY_RESULT_GROUPS,
        REPEATABILITY_MODELS,
    )

    if not model_dirs:
        return

    all_exact_scores = {model: [] for model in model_dirs}
    all_fuzzy_scores = {model: [] for model in model_dirs}

    for prompt_name in REPEATABILITY_RESULT_GROUPS:
        models_directory = results_directory / prompt_name

        available_model_dirs = [
            model
            for model in model_dirs
            if (results_directory / prompt_name / model).exists()
        ]

        if not available_model_dirs:
            continue

        exact_scores, fuzzy_scores = (
            error_bar_plot.extract_accuracy(models_directory, available_model_dirs)
        )

        for model in available_model_dirs:
            all_exact_scores[model].extend(exact_scores[model])
            all_fuzzy_scores[model].extend(fuzzy_scores[model])

    # Only keep models for which data were actually collected.
    model_dirs = [model for model in model_dirs if all_exact_scores[model] and all_fuzzy_scores[model]]

    if not model_dirs:
        return

    all_exact_scores = {model: all_exact_scores[model] for model in model_dirs}
    all_fuzzy_scores = {model: all_fuzzy_scores[model] for model in model_dirs}

    exact_mean, exact_std, fuzzy_mean, fuzzy_std = (
        error_bar_plot.compute_mean_and_std(all_exact_scores, all_fuzzy_scores)
    )

    error_bar_plot.plot_overall_accuracy(
        model_dirs,
        exact_mean,
        fuzzy_mean,
        exact_std,
        fuzzy_std,
        "repeatability"
    )


def plot_exclusion_scores(results_directory):
    ExcludePathsPlot.extract_and_plot_all(results_directory)


def plot_fuzzy_score_distributions(results_directory):
    scores = BoxPlotFuzzyScore.extract_fuzzy_scores_by_prompt_and_model(results_directory)
    BoxPlotFuzzyScore.plot_fuzzy_scores_by_prompt_and_model(scores)


def plot_accuracy_vs_document_size(results_directory, text_directory):
    field_count_data = ScatterPlot.extract_fuzzy_scores_and_field_counts(results_directory, BASELINE_EXCLUSION)

    for data_point_set in field_count_data.data_point_sets.values():
        ScatterPlot.plot_fuzzy_accuracy_vs_field_count(data_point_set)

    text_length_data = (
        ScatterPlot.extract_fuzzy_accuracy_and_text_length(
            results_directory,
            text_directory,
            BASELINE_EXCLUSION,
        )
    )

    for data_point_set in text_length_data.data_point_sets.values():
        ScatterPlot.plot_fuzzy_accuracy_vs_text_length(data_point_set)


def plot_error_types(results_directory):
    error_type_plot = ErrorTypePlot()

    error_counts = error_type_plot.extract_and_count_errors(results_directory)

    for (prompt_name, model_name), counts in error_counts.items():
        error_type_plot.plot_error_counts(prompt_name, model_name, counts)

    prompt_names = PROMPT_SELECTION_PROMPTS

    prompt_selection_model_dirs = (
        get_available_model_dirs_for_prompts(
            results_directory,
            prompt_names,
            PROMPT_SELECTION_MODELS
        )
    )

    if not prompt_selection_model_dirs:
        return

    total_error_counts = (
        error_type_plot.extract_and_count_total_errors_per_model_per_prompt_type(
            results_directory,
            prompt_selection_model_dirs,
            prompt_names,
        )
    )

    error_type_plot.plot_total_error_counts_by_prompt_type(total_error_counts)


def plot_structured_output_comparison(results_directory):
    for model in MAIN_COMPARISON_MODELS:
        prompts = list(COMMON_STRUCTURED_OUTPUT_PROMPTS)

        if model.provider == "openai":
            prompts += OPENAI_ONLY_STRUCTURED_OUTPUT_PROMPTS

        available_prompts = [
            prompt for prompt in prompts
            if (results_directory / prompt / model.result_dir / "scores_json" / "results.json").exists()
        ]

        if not available_prompts:
            continue

        structured_output_retrieval = (
            GeneralStructuredOutputRetrieval(
                results_directory=results_directory,
                prompts=available_prompts,
                model=model.result_dir,
            )
        )

        generation_successes = structured_output_retrieval.get_generation_successes()
        structured_output_retrieval.plot_successful_json_processings(generation_successes)
        accuracy_scores = structured_output_retrieval.get_accuracy_scores()
        structured_output_retrieval.plot_accuracy(accuracy_scores)

def plot_structured_output_costs(results_directory):
    cost_plot = CostPlot()
    prompts = COMMON_STRUCTURED_OUTPUT_PROMPTS + OPENAI_ONLY_STRUCTURED_OUTPUT_PROMPTS
    costs_by_method = {}

    for prompt_name in prompts:
        models_directory = results_directory / prompt_name

        model_dirs = get_available_model_dirs(results_directory, prompt_name, OPENAI_MODELS)

        if not model_dirs:
            continue

        model_names, costs = cost_plot.calculate_costs(models_directory, model_dirs)
        costs_by_method[prompt_name] = dict(zip(model_names, costs))

    if not costs_by_method:
        return

    cost_plot.plot_costs_by_method(costs_by_method)

def plot_reasoning_accuracy(results_directory):
    overall_accuracy = OverallAccuracy()
    baseline_directory = results_directory / REASONING_BASELINE_RESULT_GROUP

    reasoning_directory = results_directory / REASONING_ENABLED_RESULT_GROUP

    model_dirs = [
        baseline_config.result_dir
        for (baseline_config, reasoning_config) in REASONING_MODEL_PAIRS
        if (baseline_directory / baseline_config.result_dir / "scores_json" / "results.json").exists()
           and (reasoning_directory / reasoning_config.result_dir / "scores_json" / "results.json").exists()
    ]

    if not model_dirs:
        return

    (baseline_models, baseline_exact, baseline_fuzzy) = overall_accuracy.extract_overall_accuracy(
        baseline_directory,
        model_dirs,
    )

    (reasoning_models, reasoning_exact, reasoning_fuzzy) = overall_accuracy.extract_overall_accuracy(
        reasoning_directory,
        model_dirs,
    )

    baseline_exact_by_model = dict(zip(baseline_models, baseline_exact))
    baseline_fuzzy_by_model = dict(zip(baseline_models, baseline_fuzzy))
    reasoning_exact_by_model = dict(zip(reasoning_models, reasoning_exact))
    reasoning_fuzzy_by_model = dict(zip(reasoning_models, reasoning_fuzzy))

    common_models = [
        model for model in model_dirs
        if model in baseline_exact_by_model and model in reasoning_exact_by_model
    ]

    if not common_models:
        return

    ReasoningPlot.plot_accuracy(
        model_dirs=common_models,
        baseline_scores=[baseline_exact_by_model[model] for model in common_models],
        reasoning_scores=[reasoning_exact_by_model[model] for model in common_models],
        score_name="Exact Accuracy",
        output_name="reasoning_exact_accuracy"
    )

    ReasoningPlot.plot_accuracy(
        model_dirs=common_models,
        baseline_scores=[baseline_fuzzy_by_model[model] for model in common_models],
        reasoning_scores=[reasoning_fuzzy_by_model[model] for model in common_models],
        score_name="Fuzzy Accuracy",
        output_name="reasoning_fuzzy_accuracy"
    )


def main():
    results_directory, text_directory = get_directories()

    # Prompt-selection experiment
    plot_prompt_selection_accuracy(results_directory)
    plot_prompt_selection_runtime(results_directory)

    # plot costs
    plot_main_model_costs(results_directory)
    plot_structured_output_costs(results_directory)

    # Main model comparison
    plot_main_model_accuracy(results_directory)
    plot_main_model_runtime(results_directory)

    # Reasoning ablation
    plot_reasoning_accuracy(results_directory)

    # Repeatability
    plot_repeated_run_accuracy(results_directory)

    # Detailed analyses
    plot_exclusion_scores(results_directory)
    plot_fuzzy_score_distributions(results_directory)
    plot_accuracy_vs_document_size(results_directory, text_directory)
    plot_error_types(results_directory)

    # Structured-output comparison
    plot_structured_output_comparison(results_directory)


if __name__ == '__main__':
    main()