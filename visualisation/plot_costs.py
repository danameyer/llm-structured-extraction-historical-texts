import json
import os
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from matplotlib import pyplot as plt


class CostPlot:
    def __init__(self):
        pass

    @staticmethod
    def calculate_costs(models_directory):
        model_names = []
        costs = []
        for model_folder in os.listdir(models_directory):
            if os.path.isdir(os.path.join(models_directory, model_folder)) and model_folder in {'model_gpt-4o', 'model_gpt-4o-mini', 'model_gpt-3.5-turbo'}:
                costs_directory = os.path.join(models_directory, model_folder, 'costs')
                if os.path.exists(costs_directory):
                    json_files = [file for file in os.listdir(costs_directory) if file.endswith('.json')]
                    cost_list = list()
                    for json_file in json_files:
                        json_file_path = os.path.join(costs_directory, json_file)
                        with open(json_file_path, 'r') as file:
                            data = json.load(file)
                            cost = data['costs']
                            cost_list.append(cost)
                    total_cost = sum(cost_list)
                    model_names.append(model_folder)
                    costs.append(total_cost)
        return model_names, costs

    @staticmethod
    def plot_costs(model_names, costs, prompt_name, y_max_scale: int = None):
        x = np.arange(len(model_names))
        bar_width = 0.35
        fig, ax = plt.subplots()
        ax.bar(x, costs, bar_width, zorder=3)
        ax.set_xlabel('Models')
        ax.set_ylabel('Costs in dollar')
        # ax.set_title(f'Costs per model {prompt_name}', pad=20)
        ax.set_xticks(x)
        # truncated_model_names = [
        #     name.split("-2024-07-18")[0] if "-2024-07-18" in name else name for name in
        #     model_names]
        truncated_model_names = [name.replace("model_", "") for name in model_names]
        ax.set_xticklabels(truncated_model_names, rotation=45, ha='right')

        plt.tight_layout()
        if y_max_scale is not None:
            plt.ylim(0, y_max_scale)
        plt.grid(True, linestyle='--', alpha=0.7, zorder=0)

        output_dir = os.path.join('cost_plots')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f'cost_{prompt_name}.png'), bbox_inches='tight')
        plt.close()


if __name__ == '__main__':
    cost_plot = CostPlot()
    load_dotenv()
    base_dir = Path(os.getenv('PROJECT_BASE_DIR'))

    results_dir = os.path.join(base_dir, 'evaluation_results', 'results')
    prompt_dirs = os.listdir(results_dir)

    for prompt_dir_name in prompt_dirs:
        models_dir = os.path.join(base_dir, 'evaluation_results', 'results', prompt_dir_name)
        model_names, costs = cost_plot.calculate_costs(models_dir)
        cost_plot.plot_costs(model_names, costs, prompt_dir_name)
