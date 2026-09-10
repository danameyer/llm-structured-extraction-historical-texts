import json
import os
import re
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
                                        extracted_attributes = ', '.join(
                                            ', '.join(re.findall(r"\['(.*?)'\]",
                                                                 path if isinstance(path, str) else ' '.join(
                                                                     [p for sublist in path for p in sublist])))
                                            for path in exclude_path
                                        )

                                        # Process exact deepdiff comparison results
                                        # for comparison_result in doc['deepdiff_comparison_results']:
                                        #     for key in ['values_changed', 'iterable_item_added',
                                        #                 'iterable_item_removed', 'type_changes']:
                                        #        error_counts[f'{exclude_path}_exact'][key] += comparison_result[key]

                                        # Process fuzzy deepdiff comparison results
                                        for comparison_result in doc['deep_diff_comparison_results_fuzzy']:
                                            for key in ['values_changed', 'iterable_item_added',
                                                        'iterable_item_removed', 'type_changes']:
                                                error_counts[f'{extracted_attributes}'][key] += comparison_result[key]

                                    all_error_counts[(prompt_type_folder, model_folder)] = error_counts

        return all_error_counts

    @staticmethod
    def extract_and_count_total_errors_per_model_per_prompt_type(results_directory, model_list, prompt_list):
        """
        Extracts and counts total errors per model for each prompt type.

        Parameters:
            results_directory (str): Path to the results directory containing prompt type folders.

        Returns:
            dict: A nested dictionary where keys are prompt types, values are dictionaries with model names
                  and their respective total error counts.
        """
        # Dictionary to store errors per prompt type and per model
        error_counts_by_prompt_type = defaultdict(lambda: defaultdict(int))

        # Iterate through each prompt type directory in the results directory
        for prompt_type_folder in prompt_list:
            prompt_type_path = os.path.join(results_directory, prompt_type_folder)

            if os.path.isdir(prompt_type_path):
                # Iterate through each model directory in the prompt type directory
                for model_folder in os.listdir(prompt_type_path):
                    model_path = os.path.join(prompt_type_path, model_folder)

                    if os.path.isdir(model_path) and model_folder in model_list:
                        scores_json_dir = os.path.join(model_path, 'scores_json')

                        if os.path.exists(scores_json_dir):
                            json_files = [file for file in os.listdir(scores_json_dir) if file.endswith('.json')]
                            json_files.sort()

                            if json_files:
                                # Load the most recent JSON file
                                json_file_path = os.path.join(scores_json_dir, json_files[-1])
                                with open(json_file_path, 'r') as file:
                                    data = json.load(file)

                                # Initialize error count for the model in this prompt type
                                total_error_count = 0

                                if isinstance(data['individual_results'], list):
                                    for doc in data['individual_results']:
                                        # Continue if 'exclude_paths' not present or 'error' exists in doc
                                        if 'exclude_paths' not in doc or 'error' in doc:
                                            continue

                                        exclude_path = doc['exclude_paths']

                                        # Only consider cases where 'id' is the only excluded attribute
                                        if exclude_path == ["root['id']"]:
                                            # Process fuzzy deepdiff comparison results
                                            for comparison_result in doc['deep_diff_comparison_results_fuzzy']:
                                                for key in ['values_changed', 'iterable_item_added',
                                                            'iterable_item_removed', 'type_changes']:
                                                    total_error_count += comparison_result.get(key, 0)

                                # Update the error count for the model in this prompt type
                                error_counts_by_prompt_type[prompt_type_folder][model_folder] += total_error_count

        return dict(error_counts_by_prompt_type)

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

        ax.set_xlabel('Excluded Attributes')
        ax.set_ylabel('Error Count')
        # truncated_model_name = model_name.split('mini')[0] + 'mini' if 'mini' in model_name else model_name
        # ax.set_title(f'Error Count for {truncated_model_name} in {prompt_type} (fuzzy accuracy)',
        #             pad=20)

        ax.set_xticks(bar_positions + (num_error_types / 2 - 0.5) * bar_width)
        # ax.set_xticklabels(list(error_counts.keys()), rotation=45, ha='right')
        labels = ['no exclusions', 'cognomen', 'legal_relationship', 'place_of_origin', 'family_relations', 'title',
                  'profession']
        ax.set_xticklabels(list(labels), rotation=45, ha='right')

        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), title='Error Type', loc='upper left', bbox_to_anchor=(1, 1))

        plt.grid(True, linestyle='--', alpha=0.7, zorder=0)
        plt.tight_layout()

        output_dir = os.path.join('output_plots_errors', prompt_type)
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f'{model_name}_error_counts.png'), bbox_inches='tight')

        plt.close(fig)

    def plot_total_error_counts_by_prompt_type(self, error_counts_by_prompt_type):
        """
        Plots total error counts for each model, grouped by prompt type.

        Parameters:
            error_counts_by_prompt_type (dict): A nested dictionary where keys are prompt types, and values
                                                are dictionaries of model names and their total error counts.
        """
        # Iterate over each prompt type and plot error counts for models within that prompt type
        for prompt_type, error_counts_by_model in error_counts_by_prompt_type.items():
            fig, ax = plt.subplots(figsize=(12, 8))

            # Model names and their respective total error counts
            models = list(error_counts_by_model.keys())
            total_errors = list(error_counts_by_model.values())

            # Check if there's data to plot (to avoid empty plots)
            if not models or all(count == 0 for count in total_errors):
                print(f"No error counts to plot for prompt type: {prompt_type}")
                continue

            # Bar plot
            bar_positions = np.arange(len(models))
            bar_width = 0.35
            ax.bar(bar_positions, total_errors, bar_width, color='#1f77b4', zorder=3)

            # Set labels and title
            ax.set_xlabel('Models', fontsize=14)
            ax.set_ylabel('Total Error Count', fontsize=14)
            # ax.set_title(f'Total Error Count per Model for {prompt_type}', fontsize=16, pad=20)

            # Set x-ticks and rotate for better readability
            ax.set_xticks(bar_positions)
            truncated_model_names = [
                re.sub(r'^model_(\w+):([a-zA-Z0-9\-\.]+)-\d{4}-\d{2}-\d{2}.*', r'\2-\1', name)
                if ':' in name else
                re.sub(r'^model_([a-zA-Z0-9\-\.]+).*', r'\1', name)
                for name in models
            ]
            ax.set_xticklabels(truncated_model_names, rotation=45, ha='right', fontsize=12)

            # Grid and layout adjustments
            plt.grid(True, linestyle='--', alpha=0.7, zorder=0)
            plt.tight_layout()

            # Save the plot to output directory
            output_dir = 'output_plots_total_errors'
            os.makedirs(output_dir, exist_ok=True)
            plot_filename = f'total_error_counts_{prompt_type}.png'
            plt.savefig(os.path.join(output_dir, plot_filename), bbox_inches='tight')

            # Print confirmation and close the plot
            print(f"Saved plot for {prompt_type} as {plot_filename}")
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

    # Extract total error counts by model
    model_list = ['model_gpt-4o', 'model_gpt-4o-mini', 'model_gpt-3.5-turbo']
    prompt_dirs_standard = ['no_principles_base_prompt', 'all_principles_zero_shot', 'all_principles_few_shot',
                            'chain_of_thought', 'best_prompt']
    all_error_counts_by_model = errorTypePlot.extract_and_count_total_errors_per_model_per_prompt_type(models_dir,
                                                                                                       model_list,
                                                                                                       prompt_dirs_standard)

    # Plot total error counts for each model
    errorTypePlot.plot_total_error_counts_by_prompt_type(all_error_counts_by_model)
