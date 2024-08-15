import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv


class OverallAccuracy:
    def __init__(self):
        pass

    @staticmethod
    def extract_overall_accuracy(models_directory):
        model_names = []
        exact_scores = []
        fuzzy_scores = []

        for model_folder in os.listdir(models_directory):
            if os.path.isdir(os.path.join(models_directory, model_folder)) and model_folder.startswith('model_'):
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
    def plot_overall_accuracy(model_names, exact_scores, fuzzy_scores):
        x = np.arange(len(model_names))
        bar_width = 0.35
        fig, ax = plt.subplots()
        ax.bar(x - bar_width / 2, exact_scores, bar_width, label='Exact Score')
        ax.bar(x + bar_width / 2, fuzzy_scores, bar_width, label='Fuzzy Score')
        ax.set_xlabel('Models')
        ax.set_ylabel('Scores')
        ax.set_title('Exact vs Fuzzy Scores by Model (Excluding root["id"])')
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.legend()

        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    overall_accuracy = OverallAccuracy()
    load_dotenv()
    base_dir = Path(os.getenv('PROJECT_BASE_DIR'))
    models_dir = os.path.join(base_dir, 'evaluation_results', 'results', 'best_prompt')
    model_names, exact_scores, fuzzy_scores = overall_accuracy.extract_overall_accuracy(models_dir)
    overall_accuracy.plot_overall_accuracy(model_names, exact_scores, fuzzy_scores)
