import os
import matplotlib.pyplot as plt
import json
from dotenv import load_dotenv


class PlotMistralMetrics:
    @staticmethod
    def extract_metrics_data(root_path):
        """
        Extracts metrics data from JSON files in the specified directory structure.

        Args:
            root_path (str): The root path containing the fine_tuning_files structure.

        Returns:
            list of tuples: Each tuple contains the model name and the extracted metrics data.
        """
        all_data = []

        for folder_name in os.listdir(root_path):
            if folder_name.startswith('model') and os.path.isdir(os.path.join(root_path, folder_name)):
                model_folder_path = os.path.join(root_path, folder_name, 'metrics')

                if os.path.exists(model_folder_path) and os.path.isdir(model_folder_path):
                    for file_name in os.listdir(model_folder_path):
                        file_path = os.path.join(model_folder_path, file_name)

                        if file_name.endswith('.txt'):
                            try:
                                with open(file_path, 'r') as f:
                                    data = json.loads(f.read())
                                    checkpoints = data.get('checkpoints', [])
                                    if checkpoints:
                                        model_name = data.get('fine_tuned_model', 'unknown_model')
                                        all_data.append((model_name, checkpoints, file_name))
                            except Exception as e:
                                print(f"Error reading {file_path}: {e}")
                                continue

        return all_data

    @staticmethod
    def plot_metrics(model_name, checkpoints, output_dir, file_name):
        """
        Plots the metrics data and saves the plot to the specified output directory.

        Args:
            model_name (str): The name of the model (used for saving the plot).
            checkpoints (list): List of checkpoints containing metrics data.
            output_dir (str): The directory where plots will be saved.
            file_name (str): The original name of the file to be used in the plot file name.
        """
        # Extract data from checkpoints
        steps = [checkpoint['step_number'] for checkpoint in checkpoints]
        train_loss = [checkpoint['metrics']['train_loss'] for checkpoint in checkpoints]
        valid_loss = [checkpoint['metrics']['valid_loss'] for checkpoint in checkpoints]
        valid_mean_token_accuracy = [checkpoint['metrics']['valid_mean_token_accuracy'] for checkpoint in checkpoints]

        # Create the plots
        plt.figure(figsize=(14, 8))

        title_name = os.path.splitext(file_name)[0].replace("result_file_", "")
        plt.suptitle(f"Fine-Tuning Metrics for {title_name}", fontsize=16)

        # Determine whether to use scatter plot or line plot
        plot_type = 'scatter' if len(steps) == 1 else 'plot'

        # Plot Train Loss
        plt.subplot(2, 2, 1)
        if plot_type == 'scatter':
            plt.scatter(steps, train_loss, label='Train Loss', color='blue')
        else:
            plt.plot(steps, train_loss, label='Train Loss', color='blue')
        plt.xlabel('Step')
        plt.ylabel('Loss')
        plt.title('Train Loss Over Steps', pad=20)
        plt.grid(True)

        # Plot Validation Loss
        plt.subplot(2, 2, 2)
        if plot_type == 'scatter':
            plt.scatter(steps, valid_loss, label='Validation Loss', color='red')
        else:
            plt.plot(steps, valid_loss, label='Validation Loss', color='red')
        plt.xlabel('Step')
        plt.ylabel('Loss')
        plt.title('Validation Loss Over Steps')
        plt.grid(True)

        # Plot Validation Mean Token Accuracy
        plt.subplot(2, 2, 3)
        if plot_type == 'scatter':
            plt.scatter(steps, valid_mean_token_accuracy, label='Validation Mean Token Accuracy', color='orange')
        else:
            plt.plot(steps, valid_mean_token_accuracy, label='Validation Mean Token Accuracy', color='orange')
        plt.xlabel('Step')
        plt.ylabel('Accuracy')
        plt.title('Validation Mean Token Accuracy Over Steps')
        plt.grid(True)

        plt.tight_layout()

        os.makedirs(output_dir, exist_ok=True)

        plot_file_name = f"{os.path.splitext(file_name)[0]}_metrics_plot.png"
        plot_file_path = os.path.join(output_dir, plot_file_name)
        plt.savefig(plot_file_path)
        plt.close()

        print(f"Plot saved for {model_name} as {plot_file_path}")


def run():
    load_dotenv()
    base_dir = os.getenv('PROJECT_BASE_DIR')
    relative_path = 'fine_tuning_files/mistral/fine_tuning_evaluation'
    root_path = os.path.join(base_dir, relative_path)
    plot_mistral_metrics = PlotMistralMetrics()

    extracted_data = plot_mistral_metrics.extract_metrics_data(root_path)

    for model_name, checkpoints, file_name in extracted_data:
        plot_output_dir = os.path.join(base_dir, 'visualisation', 'mistral_ft_plots')
        plot_mistral_metrics.plot_metrics(model_name, checkpoints, plot_output_dir, file_name)


if __name__ == "__main__":
    run()
