import json
import os
from pathlib import Path

from dotenv import load_dotenv

from visualisation.box_plot_fuzzy_score import BoxPlotFuzzyScore
from visualisation.general_structured_output_retrieval import GeneralStructuredOutputRetrival
from visualisation.plot_accuracy_document_size import ScatterPlot
from visualisation.plot_costs import CostPlot
from visualisation.plot_mistral_metrics import PlotMistralMetrics
from visualisation.plot_open_ai_metrics import PlotOpenAIMetrics
from visualisation.plot_runtime import RuntimePlot
from visualisation.plot_with_error_bars import PlotWithErrorBars
from visualisation.scores_error_types import ErrorTypePlot
from visualisation.scores_for_exclude_paths import ExcludePathsPlot
from visualisation.scores_overall_accuracy import OverallAccuracy


def plot_overall_accuracy():
    overall_accuracy = OverallAccuracy()
    load_dotenv()
    base_dir = Path(os.getenv('PROJECT_BASE_DIR'))

    results_dir = os.path.join(base_dir, 'evaluation_results', 'results')

    # plots for standard models all prompt types
    prompt_dirs_standard = ['no_principles_base_prompt', 'all_principles_zero_shot', 'all_principles_few_shot', 'chain_of_thought', 'best_prompt']
    models_list_standard = ['model_gpt-4o', 'model_gpt-4o-mini', 'model_gpt-3.5-turbo']

    for prompt_dir_name in prompt_dirs_standard:
        models_dir = os.path.join(base_dir, 'evaluation_results', 'results', prompt_dir_name)
        model_names, exact_scores, fuzzy_scores = overall_accuracy.extract_overall_accuracy(models_dir, models_list_standard)
        overall_accuracy.plot_overall_accuracy(model_names, exact_scores, fuzzy_scores, prompt_dir_name)

    # plots for fine-tuned gpt-4o-mini models and standard gpt-4o-mini model
    models_list_ft = ['model_gpt-4o-mini', 'model_ft:gpt-4o-mini-2024-07-18:university-of-bielefeld::A5quE69s']
    prompt_dirs_ft = ['fine_tuning_best_prompt_fold_2']

    for prompt_dir_name in prompt_dirs_ft:
        models_dir = os.path.join(base_dir, 'evaluation_results', 'results', prompt_dir_name)
        model_names, exact_scores, fuzzy_scores = overall_accuracy.extract_overall_accuracy(models_dir,
                                                                                            models_list_ft)
        overall_accuracy.plot_overall_accuracy(model_names, exact_scores, fuzzy_scores, prompt_dir_name)


def plot_error_types():
    load_dotenv()
    base_dir = Path(os.getenv('PROJECT_BASE_DIR'))
    models_dir = os.path.join(base_dir, 'evaluation_results', 'results')

    error_type_plot = ErrorTypePlot()

    # Extract error counts
    all_error_counts = error_type_plot.extract_and_count_errors(models_dir)

    # Plot error counts for each model and prompt type
    for (prompt_type, model_name), error_counts in all_error_counts.items():
        error_type_plot.plot_error_counts(prompt_type, model_name, error_counts)


def plot_exclude_paths():
    exclude_paths_plot = ExcludePathsPlot()
    load_dotenv()
    base_dir = Path(os.getenv('PROJECT_BASE_DIR'))
    models_dir = os.path.join(base_dir, 'evaluation_results', 'results')
    exclude_paths_plot.extract_and_plot_all(models_dir)


def plot_size_vs_accuracy():
    load_dotenv()
    base_dir = Path(os.getenv('PROJECT_BASE_DIR'))
    models_dir = os.path.join(base_dir, 'evaluation_results', 'results')

    scatterPlot = ScatterPlot()

    # Extract error counts
    data_point_set_library = scatterPlot.extract_fuzzy_scores_and_field_counts(models_dir,
                                                                               ["root['id']"])

    # Plot error counts for each model and prompt type
    for key in data_point_set_library.data_point_sets:
        data_points_for_model_and_prompt = data_point_set_library.data_point_sets[key]
        scatterPlot.plot_fuzzy_accuracy_vs_field_count(data_points_for_model_and_prompt)

    data_point_set_library_text_length = scatterPlot.extract_fuzzy_accuracy_and_text_length(models_dir,
                                                                                            ["root['id']"])

    for key in data_point_set_library_text_length.data_point_sets:
        data_points_for_model_and_prompt_text_length = data_point_set_library_text_length.data_point_sets[key]
        scatterPlot.plot_fuzzy_accuracy_vs_text_length(data_points_for_model_and_prompt_text_length)


def plot_box_plot():
    box_plot_fuzzy_score = BoxPlotFuzzyScore()
    load_dotenv()
    base_dir = Path(os.getenv('PROJECT_BASE_DIR'))
    models_dir = os.path.join(base_dir, 'evaluation_results', 'results')
    all_fuzzy_scores = box_plot_fuzzy_score.extract_fuzzy_scores_by_prompt_and_model(models_dir)
    box_plot_fuzzy_score.plot_fuzzy_scores_by_prompt_and_model(all_fuzzy_scores)


def plot_runtime():
    runtime_plot = RuntimePlot()
    load_dotenv()
    base_dir = Path(os.getenv('PROJECT_BASE_DIR'))

    results_dir = os.path.join(base_dir, 'evaluation_results', 'results')
    prompt_dirs = os.listdir(results_dir)

    for prompt_dir_name in prompt_dirs:
        models_dir = os.path.join(base_dir, 'evaluation_results', 'results', prompt_dir_name)
        model_names, runtimes = runtime_plot.calculate_runtime(models_dir)
        runtime_plot.plot_runtime(model_names, runtimes, prompt_dir_name)


def plot_costs_per_model():
    cost_plot = CostPlot()
    load_dotenv()
    base_dir = Path(os.getenv('PROJECT_BASE_DIR'))

    results_dir = os.path.join(base_dir, 'evaluation_results', 'results')
    prompt_dirs = os.listdir(results_dir)

    for prompt_dir_name in prompt_dirs:
        models_dir = os.path.join(base_dir, 'evaluation_results', 'results', prompt_dir_name)
        model_names, costs = cost_plot.calculate_costs(models_dir)
        cost_plot.plot_costs(model_names, costs, prompt_dir_name)


def plot_openai_results():
    load_dotenv()
    base_dir = os.getenv('PROJECT_BASE_DIR')
    relative_path = 'fine_tuning_files/openai/fine_tuning_evaluation'
    root_path = os.path.join(base_dir, relative_path)
    plot_openai_metrics = PlotOpenAIMetrics()

    extracted_data = plot_openai_metrics.extract_metrics_data(root_path)

    for model_name, df, file_name in extracted_data:
        plot_output_dir = os.path.join(base_dir, 'visualisation', 'openai_ft_plots')
        plot_openai_metrics.plot_metrics(model_name, df, plot_output_dir, file_name)


def plot_mistral_results():
    load_dotenv()
    base_dir = os.getenv('PROJECT_BASE_DIR')
    relative_path = 'fine_tuning_files/mistral/fine_tuning_evaluation'
    root_path = os.path.join(base_dir, relative_path)
    plot_mistral_metrics = PlotMistralMetrics()

    extracted_data = plot_mistral_metrics.extract_metrics_data(root_path)

    for model_name, checkpoints, file_name in extracted_data:
        plot_output_dir = os.path.join(base_dir, 'visualisation', 'mistral_ft_plots')
        plot_mistral_metrics.plot_metrics(model_name, checkpoints, plot_output_dir, file_name)


def plot_with_error_bars():
    overall_accuracy = PlotWithErrorBars
    load_dotenv()
    base_dir = Path(os.getenv('PROJECT_BASE_DIR'))

    results_dir = os.path.join(base_dir, 'evaluation_results', 'results')
    prompt_dirs = ['best_prompt_run_1', 'best_prompt_run_2', 'best_prompt_run_3']
    model_list = ['model_gpt-4o', 'model_gpt-4o-mini', 'model_gpt-3.5-turbo']

    # Create empty dictionaries to store scores across runs
    all_exact_scores = {model: [] for model in model_list}
    all_fuzzy_scores = {model: [] for model in model_list}

    # Collect scores from each run
    for prompt_dir_name in prompt_dirs:
        models_dir = os.path.join(base_dir, 'evaluation_results', 'results', prompt_dir_name)
        exact_scores, fuzzy_scores = overall_accuracy.extract_accuracy(models_dir, model_list)

        # Append the scores for this run to the cumulative lists
        for model in model_list:
            all_exact_scores[model].extend(exact_scores[model])
            all_fuzzy_scores[model].extend(fuzzy_scores[model])

    # Compute per-model mean and standard deviations for error bars
    exact_mean, exact_std, fuzzy_mean, fuzzy_std = overall_accuracy.compute_mean_and_std(all_exact_scores, all_fuzzy_scores)

    # Plot with error bars per model
    overall_accuracy.plot_overall_accuracy(model_list, exact_mean, fuzzy_mean, exact_std, fuzzy_std, 'best_prompt')


def plot_structured_outputs_retrieval():
    load_dotenv()
    base_dir = Path(os.getenv('PROJECT_BASE_DIR'))
    evaluation_results_dir = os.path.join(base_dir, "evaluation_results")
    gsor = GeneralStructuredOutputRetrival(
        model="model_gpt-4o-2024-08-06",
        evaluation_result_path=evaluation_results_dir,
        prompts=[
            "best_prompt_no_function_calling",
            "best_prompt_structured_outputs_disabled",
            "best_prompt_structured_outputs_enabled"
        ])

    print("\n\nThese are the scores which can be plotted against one another:")
    comparable_scores = gsor.get_comparable_scores()
    print(json.dumps(comparable_scores, indent=4))

    print("\n\nThese are the mean scores:")
    mean_scores = gsor.calc_mean_scores(comparable_scores)
    print(json.dumps(mean_scores, indent=4))

    print("\n\nThese are the number of valid json files per prompt:")
    valid_files_per_prompt = gsor.get_number_of_valid_files_per_prompt()
    print(json.dumps(valid_files_per_prompt, indent=4))

    gsor.plot_successful_json_processings(valid_files_per_prompt)
    gsor.plot_mean_accuracy_for_jsons(mean_scores)


if __name__ == '__main__':
    plot_overall_accuracy()
    plot_error_types()
    plot_exclude_paths()
    plot_size_vs_accuracy()
    plot_box_plot()
    plot_costs_per_model()
    plot_runtime()
    plot_openai_results()
    plot_mistral_results()
    plot_with_error_bars()
    plot_structured_outputs_retrieval()
