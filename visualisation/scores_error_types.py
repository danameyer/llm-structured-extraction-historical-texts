import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from dotenv import load_dotenv


class ErrorTypePlot:
    def __init__(self):
        pass

    @staticmethod
    def extract_and_count_errors(results_directory):
        all_error_counts = {}

        # Iterate through each prompt type directory in the results directory
        for prompt_type_folder in os.listdir(results_directory):
            prompt_type_path = os.path.join(results_directory, prompt_type_folder)
            if os.path.isdir(prompt_type_path):
                # Iterate through each model directory in the prompt type directory
                for model_folder in os.listdir(prompt_type_path):
                    model_path = os.path.join(prompt_type_path, model_folder)
                    if os.path.isdir(model_path) and model_folder.startswith('model_'):
                        scores_json_dir = os.path.join(model_path, 'scores_json')
                        if os.path.exists(scores_json_dir):
                            json_files = [file for file in os.listdir(scores_json_dir) if file.endswith('.json')]
                            json_files.sort()

                            if json_files:
                                json_file_path = os.path.join(scores_json_dir, json_files[-1])
                                with open(json_file_path, 'r') as file:
                                    data = json.load(file)

                                if isinstance(data['individual_results'], list):
                                    error_counts = defaultdict(lambda: defaultdict(int))

                                    for doc in data['individual_results']:
                                        if 'exclude_paths' not in doc or 'error' in doc:
                                            continue
                                        exclude_path = doc['exclude_paths']

                                        # Process exact deepdiff comparison results
                                        for comparison_result in doc['deepdiff_comparison_results']:
                                            for key in ['values_changed', 'iterable_item_added',
                                                        'iterable_item_removed', 'type_changes']:
                                                error_counts[f'{exclude_path}_exact'][key] += comparison_result[key]

                                        # Process fuzzy deepdiff comparison results
                                        for comparison_result in doc['deep_diff_comparison_results_fuzzy']:
                                            for key in ['values_changed', 'iterable_item_added',
                                                        'iterable_item_removed', 'type_changes']:
                                                error_counts[f'{exclude_path}_fuzzy'][key] += comparison_result[key]

                                    all_error_counts[(prompt_type_folder, model_folder)] = error_counts

        return all_error_counts

    def plot_error_counts(self, prompt_type, model_name, error_counts):
        error_types = ['values_changed', 'iterable_item_added', 'iterable_item_removed', 'type_changes']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Assign colors to each error type
        fig, ax = plt.subplots(figsize=(12, 8))

        bar_width = 0.2
        num_error_types = len(error_types)
        num_combinations = len(error_counts)

        bar_positions = np.arange(num_combinations) * (num_error_types + 1) * bar_width

        for i, (exclude_path, scores) in enumerate(error_counts.items()):
            for j, error_type in enumerate(error_types):
                count = scores.get(error_type, 0)
                ax.bar(bar_positions[i] + j * bar_width, count, width=bar_width, color=colors[j],
                       label=error_type if i == 0 else "")

        ax.set_xlabel('Exclusion Path and Type')
        ax.set_ylabel('Count of Errors')
        ax.set_title(f'Error Count for {model_name} in {prompt_type}')

        ax.set_xticks(bar_positions + (num_error_types / 2 - 0.5) * bar_width)
        ax.set_xticklabels(list(error_counts.keys()), rotation=45, ha='right')

        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), title='Error Type')

        plt.tight_layout()

        output_dir = os.path.join('output_plots_errors', prompt_type)
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f'{model_name}_error_counts.png'))

        plt.close(fig)


if __name__ == '__main__':
    load_dotenv()
    base_dir = Path(os.getenv('PROJECT_BASE_DIR'))
    models_dir = os.path.join(base_dir, 'evaluation_results', 'results')

    errorTypePlot = ErrorTypePlot()

    # Extract error counts
    all_error_counts = errorTypePlot.extract_and_count_errors(models_dir)

    # Plot error counts for each model and prompt type
    for (prompt_type, model_name), error_counts in all_error_counts.items():
        errorTypePlot.plot_error_counts(prompt_type, model_name, error_counts)
