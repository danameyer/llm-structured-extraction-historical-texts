import json
import os
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from matplotlib import pyplot as plt


class PlotWithErrorBars:
    def __init__(self):
        pass

    @staticmethod
    def extract_accuracy(models_directory, model_list):
        exact_scores = {model: [] for model in model_list}
        fuzzy_scores = {model: [] for model in model_list}

        for model_folder in os.listdir(models_directory):
            if os.path.isdir(os.path.join(models_directory, model_folder)) and model_folder in model_list:
                # Adjust the path to look into the scores_json folder
                scores_json_dir = os.path.join(models_directory, model_folder, 'scores_json')

                # Ensure the scores_json directory exists
                if os.path.exists(scores_json_dir):
                    json_files = [file for file in os.listdir(scores_json_dir) if file.endswith('.json')]

                    # Sort files by timestamp in the filename if necessary
                    json_files.sort()
                    if json_files:
                        json_file_path = os.path.join(scores_json_dir, json_files[-1])

                        with open(json_file_path, 'r') as file:
                            data = json.load(file)

                        # Ensure overall_results is a list
                        if isinstance(data['overall_results'], list):
                            # Filter to get only the results for "overall_path_exclude": "root['id']"
                            for result in data['overall_results']:
                                if isinstance(result, dict) and result.get('overall_path_exclude') == "root['id']":
                                    exact_scores[model_folder].append(result['overall_exact_score'])
                                    fuzzy_scores[model_folder].append(result['overall_fuzzy_score'])
                                    break  # Assuming we only need one entry per model
                else:
                    print(f"No scores_json directory found in {model_folder}")

        return exact_scores, fuzzy_scores

    @staticmethod
    def compute_mean_and_std(exact_scores, fuzzy_scores):
        # Calculate mean and standard deviation per model across runs
        exact_mean = {model: np.mean(scores) for model, scores in exact_scores.items()}
        exact_std = {model: np.std(scores) for model, scores in exact_scores.items()}
        fuzzy_mean = {model: np.mean(scores) for model, scores in fuzzy_scores.items()}
        fuzzy_std = {model: np.std(scores) for model, scores in fuzzy_scores.items()}

        return exact_mean, exact_std, fuzzy_mean, fuzzy_std

    @staticmethod
    def plot_overall_accuracy(model_names, exact_mean, fuzzy_mean, exact_std, fuzzy_std, prompt_name):
        x = np.arange(len(model_names))
        bar_width = 0.35

        fig, ax = plt.subplots()

        # Convert dictionaries to lists for plotting
        exact_mean_values = [exact_mean[model] for model in model_names]
        fuzzy_mean_values = [fuzzy_mean[model] for model in model_names]
        exact_std_values = [exact_std[model] for model in model_names]
        fuzzy_std_values = [fuzzy_std[model] for model in model_names]

        # Plot with error bars per model for exact and fuzzy scores
        ax.bar(x - bar_width / 2, exact_mean_values, bar_width, yerr=exact_std_values, label='Exact Accuracy', zorder=3,
               capsize=5)
        ax.bar(x + bar_width / 2, fuzzy_mean_values, bar_width, yerr=fuzzy_std_values, label='Fuzzy Accuracy', zorder=3,
               capsize=5)

        ax.set_xlabel('Models')
        ax.set_ylabel('Accuracy')
        # ax.set_title(f'Exact vs Fuzzy Accuracy by Model for {prompt_name}', pad=20)
        ax.set_xticks(x)

        # Truncate model names for better visibility
        truncated_model_names = [name.replace("model_", "") for name in model_names]
        ax.set_xticklabels(truncated_model_names, rotation=45, ha='right')
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

        plt.tight_layout()
        plt.grid(True, linestyle='--', alpha=0.7, zorder=0)
        plt.ylim(0.0, 1.0)

        # Save the plot
        output_dir = os.path.join('output_error_bar_plots')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f'overall_scores_error_bars_{prompt_name}.png'), bbox_inches='tight')


if __name__ == '__main__':
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
