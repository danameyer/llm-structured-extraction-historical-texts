import json
import os
import re
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv
from matplotlib import pyplot as plt


class BoxPlotFuzzyScore:

    @staticmethod
    def extract_fuzzy_scores_by_prompt_and_model(models_directory):
        """
        Extracts fuzzy scores from JSON files in a directory structure, grouped by prompt type and model.

        Args:
            models_directory (str): The path to the root directory containing model results.

        Returns:
            dict: A nested dictionary where the first key is the prompt type, the second key is the model name,
                  and the value is a dictionary with exclude paths as keys and fuzzy accuracy scores as values.
        """
        scores_by_prompt_and_model = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

        for prompt_type_folder in os.listdir(models_directory):
            prompt_type_path = os.path.join(models_directory, prompt_type_folder)
            if os.path.isdir(prompt_type_path):
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
                                    # Extract and add fuzzy scores to the list
                                    for result in data['individual_results']:
                                        exclude_paths = tuple(result.get('exclude_paths', []))
                                        fuzzy_score = result.get('fuzzy_score')
                                        if fuzzy_score is not None:
                                            scores_by_prompt_and_model[prompt_type_folder][model_folder][
                                                exclude_paths].append(fuzzy_score)

        return scores_by_prompt_and_model

    @staticmethod
    def plot_fuzzy_scores_by_prompt_and_model(scores_by_prompt_and_model):
        """
        Plots a box plot for the fuzzy accuracy scores grouped by exclude paths for each prompt type and model.

        Args:
            scores_by_prompt_and_model (dict): A nested dictionary where the first key is the prompt type,
                                               the second key is the model name, and the value is a dictionary
                                               with exclude paths as keys and fuzzy accuracy scores as values.
        """
        for prompt_type, models_scores in scores_by_prompt_and_model.items():
            for model_name, scores_by_exclude_paths in models_scores.items():
                labels = [
                    ', '.join([attr for path in paths for attr in re.findall(r"\['(.*?)'\]", path)])
                    for paths in scores_by_exclude_paths.keys()
                ]
                data = list(scores_by_exclude_paths.values())

                plt.figure(figsize=(10, 6))
                plt.boxplot(data, patch_artist=True, medianprops=dict(color='black'))
                plt.xticks(ticks=range(1, len(labels) + 1), labels=labels, rotation=45, ha='right')
                truncated_model_name = model_name.split('mini')[0] + 'mini' if 'mini' in model_name else model_name
                plt.title(f'Fuzzy Accuracy Box Plot for {prompt_type} - {truncated_model_name}')
                plt.ylabel('Fuzzy Accuracy')
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.tight_layout()

                # Save the plot
                plot_filename = f'{prompt_type}_{model_name}_fuzzy_accuracy.png'
                output_dir = os.path.join('output_boxplots', prompt_type)
                os.makedirs(output_dir, exist_ok=True)
                plt.savefig(os.path.join(output_dir, plot_filename))
                plt.close()


if __name__ == '__main__':
    box_plot_fuzzy_score = BoxPlotFuzzyScore()
    load_dotenv()
    base_dir = Path(os.getenv('PROJECT_BASE_DIR'))
    models_dir = os.path.join(base_dir, 'evaluation_results', 'results')
    all_fuzzy_scores = box_plot_fuzzy_score.extract_fuzzy_scores_by_prompt_and_model(models_dir)
    box_plot_fuzzy_score.plot_fuzzy_scores_by_prompt_and_model(all_fuzzy_scores)
