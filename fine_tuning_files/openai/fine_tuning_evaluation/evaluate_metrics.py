import os
import pandas as pd
import glob

from dotenv import load_dotenv


@staticmethod
def evaluate_metrics(base_dir):
    # Use glob to find all 'metrics' folders under 'model_' directories
    metrics_paths = glob.glob(os.path.join(base_dir, 'model_*/metrics/*.txt'))

    # List to store data from each fold
    fold_dfs = []

    # Loop through each .txt file in the 'metrics' directories
    for file_path in metrics_paths:
        # Assuming your .txt is formatted like a CSV (comma-separated values)
        try:
            df = pd.read_csv(file_path, delimiter=',')
            # Append this fold's DataFrame to the list
            fold_dfs.append(df)
        except pd.errors.EmptyDataError:
            print(f"Warning: {file_path} is empty or incorrectly formatted.")

    # Check if any DataFrames were loaded
    if not fold_dfs:
        print("No valid data found in the specified directories.")
        return

    # Concatenate all fold DataFrames into a single DataFrame
    full_df = pd.concat(fold_dfs, ignore_index=True)

    # Filter out rows where 'valid_mean_token_accuracy' is not null
    valid_acc_df = full_df[full_df['valid_mean_token_accuracy'].notnull()]

    # Group by 'step' to calculate statistics per step across folds
    grouped = valid_acc_df.groupby('step')['valid_mean_token_accuracy']

    # Calculate the mean accuracy, variance, and standard deviation
    mean_accuracy = grouped.mean()
    variance = grouped.var()
    std_dev = grouped.std()

    # Create a new DataFrame with the results
    result_df = pd.DataFrame({
        'Mean Accuracy': mean_accuracy,
        'Variance': variance,
        'Standard Deviation': std_dev
    })

    overall_mean_accuracy = valid_acc_df['valid_mean_token_accuracy'].mean()
    overall_variance = valid_acc_df['valid_mean_token_accuracy'].var()
    overall_std_dev = valid_acc_df['valid_mean_token_accuracy'].std()

    # Add the overall statistics to the result DataFrame for easier comparison
    overall_stats = pd.DataFrame({
        'Mean Accuracy': [overall_mean_accuracy],
        'Variance': [overall_variance],
        'Standard Deviation': [overall_std_dev]
    }, index=['Overall'])

    # Combine the per-step and overall results
    final_df = pd.concat([result_df, overall_stats])

    # Save the results to a CSV file for easier access
    output_file = os.path.join(base_dir, 'accuracy_fluctuations_from_txt.csv')
    final_df.to_csv(output_file)

    # Print the result DataFrame to view the calculated statistics
    print(final_df)


if __name__ == '__main__':
    load_dotenv()
    base_dir = os.getenv("PROJECT_BASE_DIR")
    file_path = os.path.join(base_dir, 'fine_tuning_files', 'openai', 'fine_tuning_evaluation')

    if file_path:
        evaluate_metrics(file_path)
    else:
        print("Error: File path not set")
