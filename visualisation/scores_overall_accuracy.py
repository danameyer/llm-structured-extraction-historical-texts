import json
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv


class OverallAccuracy:
    def __init__(self):
        pass

    @staticmethod
    def extract_overall_accuracy(models_directory, models_list):
        model_names = []
        exact_scores = []
        fuzzy_scores = []

        for model_folder in os.listdir(models_directory):
            if os.path.isdir(os.path.join(models_directory, model_folder)) and model_folder in models_list:
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
                                    model_names.append(model_folder)
                                    exact_scores.append(result['overall_exact_score'])
                                    fuzzy_scores.append(result['overall_fuzzy_score'])
                                    break  # Assuming we only need one entry per model
                else:
                    print(f"No scores_json directory found in {model_folder}")

        return model_names, exact_scores, fuzzy_scores

    @staticmethod
    def plot_overall_accuracy(model_names, exact_scores, fuzzy_scores, prompt_name):
        x = np.arange(len(model_names))
        bar_width = 0.35
        fig, ax = plt.subplots()
        ax.bar(x - bar_width / 2, exact_scores, bar_width, label='Exact Accuracy', zorder=3)
        ax.bar(x + bar_width / 2, fuzzy_scores, bar_width, label='Fuzzy Accuracy', zorder=3)
        ax.set_xlabel('Models')
        ax.set_ylabel('Accuracy')
        # ax.set_title(f'Exact vs Fuzzy Accuracy by Model for {prompt_name}', pad=20)
        ax.set_xticks(x)
        truncated_model_names = [re.sub(r'(-\d{4}-\d{2}-\d{2}).*', '', name.replace("model_", "").replace(":", "_")) for
                                 name in model_names]

        ax.set_xticklabels(truncated_model_names, rotation=45, ha='right')
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

        plt.tight_layout()
        plt.grid(True, linestyle='--', alpha=0.7, zorder=0)
        plt.ylim(0.0, 1.0)
        output_dir = os.path.join('output_overall_plots')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f'overall_scores_{prompt_name}.png'), bbox_inches='tight')


if __name__ == '__main__':
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
