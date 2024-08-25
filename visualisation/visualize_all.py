import os
from pathlib import Path

from dotenv import load_dotenv

from visualisation.scores_error_types import ErrorTypePlot
from visualisation.scores_for_exclude_paths import ExcludePathsPlot
from visualisation.scores_overall_accuracy import OverallAccuracy


def plot_overall_accuracy(prompt_name):
    overall_accuracy = OverallAccuracy()
    load_dotenv()
    base_dir = Path(os.getenv('PROJECT_BASE_DIR'))
    models_dir = os.path.join(base_dir, 'evaluation_results', 'results', prompt_name)
    model_names, exact_scores, fuzzy_scores = overall_accuracy.extract_overall_accuracy(models_dir)
    overall_accuracy.plot_overall_accuracy(model_names, exact_scores, fuzzy_scores, prompt_name)


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


if __name__ == '__main__':
    plot_overall_accuracy('best_prompt')
    plot_overall_accuracy('chain_of_thought')

    plot_error_types()
    plot_exclude_paths()
