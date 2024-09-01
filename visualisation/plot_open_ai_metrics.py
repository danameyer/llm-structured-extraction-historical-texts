import os
import matplotlib.pyplot as plt
import pandas as pd
from dotenv import load_dotenv


class PlotOpenAIMetrics:
    @staticmethod
    def extract_metrics_data(root_path):
        """
        Extracts metrics data from CSV files in the specified directory structure.

        Args:
            root_path (str): The root path containing the fine_tuning_files structure.

        Returns:
            list of tuples: Each tuple contains the model name and the DataFrame of the extracted data.
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
                                df = pd.read_csv(file_path)
                                all_data.append((folder_name, df, file_name))
                            except Exception as e:
                                print(f"Error reading {file_path}: {e}")
                                continue

        return all_data

    @staticmethod
    def plot_metrics(model_name, df, output_dir, file_name):
        """
        Plots the metrics data and saves the plot to the specified output directory.

        Args:
            model_name (str): The name of the model (used for saving the plot).
            df (DataFrame): The DataFrame containing the metrics data.
            output_dir (str): The directory where plots will be saved.
            file_name (str): The original name of the file to be used in the plot file name.
        """
        # Create the plots
        plt.figure(figsize=(14, 8))

        # Plot Train Loss
        plt.subplot(2, 2, 1)
        plt.plot(df['step'], df['train_loss'], label='Train Loss', color='blue')
        plt.xlabel('Step')
        plt.ylabel('Loss')
        plt.title('Train Loss Over Steps')
        plt.grid(True)

        # Plot Train Accuracy
        plt.subplot(2, 2, 2)
        plt.plot(df['step'], df['train_accuracy'], label='Train Accuracy', color='green')
        plt.xlabel('Step')
        plt.ylabel('Accuracy')
        plt.title('Train Accuracy Over Steps')
        plt.grid(True)

        # Plot Validation Loss
        plt.subplot(2, 2, 3)
        plt.plot(df['step'], df['valid_loss'], label='Validation Loss', color='red')
        plt.xlabel('Step')
        plt.ylabel('Loss')
        plt.title('Validation Loss Over Steps')
        plt.grid(True)

        # Plot Validation Mean Token Accuracy
        plt.subplot(2, 2, 4)
        plt.plot(df['step'], df['valid_mean_token_accuracy'], label='Validation Mean Token Accuracy', color='orange')
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
    relative_path = 'fine_tuning_files/openai/fine_tuning_evaluation'
    root_path = os.path.join(base_dir, relative_path)
    plot_openai_metrics = PlotOpenAIMetrics()

    extracted_data = plot_openai_metrics.extract_metrics_data(root_path)

    for model_name, df, file_name in extracted_data:
        plot_output_dir = os.path.join(base_dir, 'visualisation', 'openai_ft_plots')
        plot_openai_metrics.plot_metrics(model_name, df, plot_output_dir, file_name)


if __name__ == "__main__":
    run()
