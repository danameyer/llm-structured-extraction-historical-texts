import json
import os
import re
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from matplotlib import pyplot as plt


class RuntimePlot:
    def __init__(self):
        pass

    @staticmethod
    def calculate_runtime(models_directory, model_list):
        model_names = []
        runtimes = []
        for model_folder in os.listdir(models_directory):
            if os.path.isdir(os.path.join(models_directory, model_folder)) and model_folder in model_list:
                runtime_directory = os.path.join(models_directory, model_folder, 'runtime')
                if os.path.exists(runtime_directory):
                    json_files = [file for file in os.listdir(runtime_directory) if file.endswith('.json')]
                    runtime_list = list()
                    for json_file in json_files:
                        json_file_path = os.path.join(runtime_directory, json_file)
                        with open(json_file_path, 'r') as file:
                            data = json.load(file)
                            runtime = data['total_runtime']
                            runtime_list.append(runtime)
                    total_runtime = sum(runtime_list)
                    model_names.append(model_folder)
                    runtimes.append(total_runtime)
        return model_names, runtimes

    @staticmethod
    def plot_runtime(model_names, runtimes, prompt_name, y_max_scale: int = None):
        x = np.arange(len(model_names))
        bar_width = 0.35
        fig, ax = plt.subplots()
        ax.bar(x, runtimes, bar_width, zorder=3)
        ax.set_xlabel('Models')
        ax.set_ylabel('Runtime in seconds')
        ax.set_xticks(x)
        truncated_model_names = [
            re.sub(r'^model_(\w+):([a-zA-Z0-9\-\.]+)-\d{4}-\d{2}-\d{2}.*', r'\2-\1', name)
            if ':' in name else
            re.sub(r'^model_([a-zA-Z0-9\-\.]+).*', r'\1', name)
            for name in model_names
        ]
        ax.set_xticklabels(truncated_model_names, rotation=45, ha='right')

        plt.tight_layout()
        if y_max_scale is not None:
            plt.ylim(0, y_max_scale)
        plt.grid(True, linestyle='--', alpha=0.7, zorder=0)

        output_dir = os.path.join('runtime_plots')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f'runtime_{prompt_name}.png'), bbox_inches='tight')
        plt.close()


if __name__ == '__main__':
    runtime_plot = RuntimePlot()
    load_dotenv()
    base_dir = Path(os.getenv('PROJECT_BASE_DIR'))

    results_dir = os.path.join(base_dir, 'evaluation_results', 'results')

    prompt_dirs_standard = ['no_principles_base_prompt', 'all_principles_zero_shot', 'all_principles_few_shot', 'chain_of_thought', 'best_prompt']
    models_list_standard = ['model_gpt-4o', 'model_gpt-4o-mini', 'model_gpt-3.5-turbo']
    for prompt_dir_name in prompt_dirs_standard:
        models_dir = os.path.join(base_dir, 'evaluation_results', 'results', prompt_dir_name)
        model_names, runtimes = runtime_plot.calculate_runtime(models_dir, models_list_standard)
        runtime_plot.plot_runtime(model_names, runtimes, prompt_dir_name)
