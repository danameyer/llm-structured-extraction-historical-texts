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


MODEL_DIRS = ['model_gpt-5.6-luna', 'model_gpt-5.6-terra', 'model_gpt-5.6-sol']
PROMPT_SELECTION_PROMPTS = [
    'chain_of_thought',
    'all_principles_zero_shot',
    'all_principles_few_shot',
    'no_principles_base_prompt',
]
BEST_PROMPT = 'best_prompt'
BEST_PROMPT_RUNS = ['best_prompt', 'best_prompt_run_2', 'best_prompt_run_3']
STRUCTURED_OUTPUT_PROMPTS = [
    'best_prompt_prompt_only_json',
    'best_prompt_direct_structured_outputs',
    'best_prompt_structured_outputs_disabled',
    'best_prompt_structured_outputs_enabled',
]
STRUCTURED_OUTPUT_MODEL = 'model_gpt-5.6-sol'
BASELINE_EXCLUSION = ["root['id']"]


def get_directories():
    load_dotenv()
    base_dir = os.getenv('PROJECT_BASE_DIR')

    if not base_dir:
        raise ValueError("PROJECT_BASE_DIR is not set.")

    base_dir = Path(base_dir)
    results_directory = base_dir / 'evaluation_results' / 'results'
    text_directory = base_dir / 'evaluation_results' / 'txt_files_function_calling_evaluation'

    return results_directory, text_directory


def plot_overall_accuracy(results_directory):
    overall_accuracy = OverallAccuracy()
    prompt_names = PROMPT_SELECTION_PROMPTS + [BEST_PROMPT]

    for prompt_name in prompt_names:
        models_directory = results_directory / prompt_name
        model_names, exact_scores, fuzzy_scores = overall_accuracy.extract_overall_accuracy(models_directory, MODEL_DIRS)
        overall_accuracy.plot_overall_accuracy(model_names, exact_scores, fuzzy_scores, prompt_name)


def plot_costs(results_directory):
    cost_plot = CostPlot()
    prompt_names = PROMPT_SELECTION_PROMPTS + [BEST_PROMPT]

    for prompt_name in prompt_names:
        models_directory = results_directory / prompt_name
        model_names, costs = cost_plot.calculate_costs(models_directory, MODEL_DIRS)
        cost_plot.plot_costs(model_names, costs, prompt_name)


def plot_runtime(results_directory):
    runtime_plot = RuntimePlot()
    prompt_names = PROMPT_SELECTION_PROMPTS + [BEST_PROMPT]

    for prompt_name in prompt_names:
        models_directory = results_directory / prompt_name
        model_names, runtimes = runtime_plot.calculate_runtime(models_directory, MODEL_DIRS)
        runtime_plot.plot_runtime(model_names, runtimes, prompt_name)


def plot_repeated_run_accuracy(results_directory):
    error_bar_plot = PlotWithErrorBars()
    all_exact_scores = {model: [] for model in MODEL_DIRS}
    all_fuzzy_scores = {model: [] for model in MODEL_DIRS}

    for prompt_name in BEST_PROMPT_RUNS:
        models_directory = results_directory / prompt_name
        exact_scores, fuzzy_scores = error_bar_plot.extract_accuracy(models_directory, MODEL_DIRS)

        for model in MODEL_DIRS:
            all_exact_scores[model].extend(exact_scores[model])
            all_fuzzy_scores[model].extend(fuzzy_scores[model])

    exact_mean, exact_std, fuzzy_mean, fuzzy_std = (
        error_bar_plot.compute_mean_and_std(all_exact_scores, all_fuzzy_scores)
    )

    error_bar_plot.plot_overall_accuracy(
        MODEL_DIRS,
        exact_mean,
        fuzzy_mean,
        exact_std,
        fuzzy_std,
        BEST_PROMPT,
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

    total_error_counts = (
        error_type_plot.extract_and_count_total_errors_per_model_per_prompt_type(
            results_directory,
            MODEL_DIRS,
            PROMPT_SELECTION_PROMPTS + [BEST_PROMPT],
        )
    )

    error_type_plot.plot_total_error_counts_by_prompt_type(total_error_counts)


def plot_structured_output_comparison(results_directory):
    structured_output_retrieval = (
        GeneralStructuredOutputRetrieval(
            results_directory=results_directory,
            prompts=STRUCTURED_OUTPUT_PROMPTS,
            model=STRUCTURED_OUTPUT_MODEL,
        )
    )

    generation_successes = structured_output_retrieval.get_generation_successes()
    structured_output_retrieval.plot_successful_json_processings(generation_successes)
    accuracy_scores = structured_output_retrieval.get_accuracy_scores()
    structured_output_retrieval.plot_accuracy(accuracy_scores)


def main():
    results_directory, text_directory = get_directories()
    plot_overall_accuracy(results_directory)
    plot_costs(results_directory)
    plot_runtime(results_directory)
    plot_repeated_run_accuracy(results_directory)
    plot_exclusion_scores(results_directory)
    plot_fuzzy_score_distributions(results_directory)
    plot_accuracy_vs_document_size(results_directory, text_directory)
    plot_error_types(results_directory)
    plot_structured_output_comparison(results_directory)


if __name__ == '__main__':
    main()