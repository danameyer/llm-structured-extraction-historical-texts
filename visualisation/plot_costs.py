import json
import re
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


class CostPlot:

    @staticmethod
    def calculate_costs(
            models_directory: str | Path,
            model_list: list[str],
    ) -> tuple[list[str], list[float]]:
        models_directory = Path(models_directory)
        model_names = []
        costs = []

        for model_folder in model_list:
            costs_directory = models_directory / model_folder / 'costs'

            if not costs_directory.is_dir():
                print(f"No costs directory found for {model_folder}")
                continue

            json_files = sorted(costs_directory.glob('*.json'))

            if not json_files:
                print(f"No cost files found for {model_folder}")
                continue

            cost_list = []

            for json_file_path in json_files:
                with open(json_file_path, 'r', encoding='utf-8') as file:
                    data = json.load(file)

                cost_list.append(data['costs'])

            model_names.append(model_folder)
            costs.append(sum(cost_list))

        return model_names, costs

    @staticmethod
    def plot_costs(
            model_names,
            costs,
            prompt_name,
            y_max_scale: float | None = None,
    ):
        x = np.arange(len(model_names))
        bar_width = 0.35
        fig, ax = plt.subplots()
        ax.bar(x, costs, bar_width, zorder=3)
        ax.set_xlabel('Models')
        ax.set_ylabel('Cost (USD)')
        ax.set_xticks(x)

        truncated_model_names = [
            re.sub(r'^model_(\w+):([a-zA-Z0-9\-.]+)-\d{4}-\d{2}-\d{2}.*', r'\2-\1', name)
            if ':' in name else
            re.sub(r'^model_([a-zA-Z0-9\-.]+).*', r'\1', name)
            for name in model_names
        ]

        ax.set_xticklabels(truncated_model_names, rotation=45, ha='right')

        if y_max_scale is not None:
            ax.set_ylim(0, y_max_scale)

        ax.grid(True, linestyle='--', alpha=0.7, zorder=0)
        plt.tight_layout()
        output_dir = Path('cost_plots')
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_dir / f'cost_{prompt_name}.png', bbox_inches='tight')
        plt.close(fig)