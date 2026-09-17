import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from visualisation.utils import format_model_name


class CostPlot:

    def __init__(self, output_directory: str | Path = "cost_plots"):
        self.output_directory = Path(output_directory)

    @staticmethod
    def calculate_costs(
            models_directory: str | Path,
            model_list: list[str],
    ) -> tuple[list[str], list[float]]:
        models_directory = Path(models_directory)
        model_names = []
        costs = []

        for model_folder in model_list:
            costs_directory = models_directory / model_folder / "costs"

            if not costs_directory.is_dir():
                print(f"No costs directory found for {model_folder}")
                continue

            json_files = sorted(costs_directory.glob("*.json"))

            if not json_files:
                print(f"No cost files found for {model_folder}")
                continue

            cost_list = []

            for json_file_path in json_files:
                with open(json_file_path, "r", encoding="utf-8") as file:
                    data = json.load(file)

                cost_list.append(data["costs"])

            model_names.append(model_folder)
            costs.append(sum(cost_list))

        return model_names, costs

    @staticmethod
    def _format_method_name(prompt_name: str) -> str:
        method_names = {
            "best_prompt_prompt_only_json":
                "Prompt-only JSON",
            "best_prompt_direct_structured_outputs":
                "JSON schema",
            "best_prompt_structured_outputs_disabled":
                "Tool calling",
            "best_prompt_structured_outputs_enabled":
                "Strict tool calling",
        }

        return method_names.get(prompt_name, prompt_name)

    def plot_costs(
            self,
            model_names,
            costs,
            prompt_name,
            y_max_scale: float | None = None,
    ):
        x = np.arange(len(model_names))
        bar_width = 0.35
        fig, ax = plt.subplots()
        ax.bar(x, costs, bar_width, zorder=3)
        ax.set_xlabel("Models")
        ax.set_ylabel("Cost (USD)")
        ax.set_xticks(x)
        display_model_names = [format_model_name(model) for model in model_names]
        ax.set_xticklabels(display_model_names, rotation=45, ha="right")

        if y_max_scale is not None:
            ax.set_ylim(0, y_max_scale)

        ax.grid(True, linestyle="--", alpha=0.7, zorder=0)
        plt.tight_layout()
        output_directory = self.output_directory / "model_comparison"
        output_directory.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_directory / f"cost_{prompt_name}.png", bbox_inches="tight")
        plt.close(fig)

    def plot_costs_by_method(self, costs_by_method):
        methods = list(costs_by_method.keys())
        model_names = sorted({model for model_costs in costs_by_method.values() for model in model_costs})

        if not model_names:
            return

        x = np.arange(len(methods))
        width = 0.8 / len(model_names)
        fig, ax = plt.subplots(figsize=(10, 6))

        for index, model_name in enumerate(model_names):
            model_costs = [costs_by_method[method].get(model_name, np.nan) for method in methods]
            positions = (x + index * width - ((len(model_names) - 1) * width / 2))
            ax.bar(positions, model_costs, width, label=format_model_name(model_name))

        ax.set_xticks(x)
        ax.set_xticklabels([self._format_method_name(method) for method in methods], rotation=20, ha="right")
        ax.set_xlabel("JSON generation method")
        ax.set_ylabel("Total cost (USD)")
        ax.set_title("Cost by JSON Generation Method")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.7, axis="y", zorder=0)
        plt.tight_layout()
        output_directory = self.output_directory / "structured_output_comparison"
        output_directory.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_directory / "cost_by_json_production_method.png", bbox_inches="tight")
        plt.close(fig)