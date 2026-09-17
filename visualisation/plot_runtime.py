import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from visualisation.utils import format_model_name


class RuntimePlot:

    @staticmethod
    def calculate_runtime(
            models_directory: str | Path,
            model_list: list[str],
    ) -> tuple[list[str], list[float]]:
        models_directory = Path(models_directory)
        model_names = []
        runtimes = []

        for model_folder in model_list:
            runtime_directory = models_directory / model_folder / 'runtime'

            if not runtime_directory.is_dir():
                print(f"No runtime directory found for {model_folder}")
                continue

            json_files = sorted(runtime_directory.glob('*.json'))

            if not json_files:
                print(f"No runtime files found for {model_folder}")
                continue

            runtime_list = []

            for json_file_path in json_files:
                with open(json_file_path, 'r', encoding='utf-8') as file:
                    data = json.load(file)

                runtime_list.append(data['total_runtime'])

            model_names.append(model_folder)
            runtimes.append(sum(runtime_list))

        return model_names, runtimes

    @staticmethod
    def plot_runtime(
            model_names,
            runtimes,
            prompt_name,
            y_max_scale: float | None = None,
    ):
        x = np.arange(len(model_names))
        bar_width = 0.35
        fig, ax = plt.subplots()
        ax.bar(x, runtimes, bar_width, zorder=3)
        ax.set_xlabel('Models')
        ax.set_ylabel('Runtime (seconds)')
        ax.set_xticks(x)

        display_model_names = [format_model_name(model) for model in model_names]
        ax.set_xticklabels(display_model_names, rotation=45, ha='right')

        if y_max_scale is not None:
            ax.set_ylim(0, y_max_scale)

        ax.grid(True, linestyle='--', alpha=0.7, zorder=0)
        plt.tight_layout()
        output_dir = Path('runtime_plots')
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_dir / f'runtime_{prompt_name}.png', bbox_inches='tight')
        plt.close(fig)