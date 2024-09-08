import json
import os
from pathlib import Path
from typing import Optional, List
import time
from dotenv import load_dotenv
from mistralai import Mistral

from fine_tuning.data_split.perform_data_split import PerformDataSplit
from fine_tuning.fine_tuning_api_interaction.create_model_registry import ModelRegistry
from fine_tuning.message_objects.mistral_message_object import MistralMessageObject

load_dotenv()
mistral_api_key: Optional[str] = os.getenv("MISTRAL_API_KEY")


class MistralApiInteractionForFineTuning:
    def __init__(self, result_file_path):
        self.client = Mistral(
            api_key=os.environ.get("MISTRAL_API_KEY"),
        )
        self.result_file_path = result_file_path
        os.makedirs(result_file_path, exist_ok=True)

    def upload_data_to_mistral_api(self, file: str):
        response = self.client.files.upload(
            file={
                "file_name": file,
                "content": open(file, "rb"),
            }
        )
        return response

    def create_model_mistral(self, train_file_id, validation_file_id, model):
        response = self.client.fine_tuning.jobs.create(
            model=model,
            training_files=[{"file_id": train_file_id, "weight": 1}],
            validation_files=[validation_file_id],
            hyperparameters={
                "training_steps": 10,
                "learning_rate": 0.0001
            },
            auto_start=False
        )
        return response

    def retrieve_fine_tuning_results_mistral(self, finetune_job_id):
        ft_results = self.client.fine_tuning.jobs.get(job_id=finetune_job_id)
        return ft_results

    def to_dict(self, obj):
        if isinstance(obj, list):
            return [self.to_dict(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: self.to_dict(value) for key, value in obj.items()}
        elif hasattr(obj, "__dict__"):
            return self.to_dict(obj.__dict__)
        else:
            return obj

    def wait_for_job_to_finish(self, finetune_job_id, model, fold_number, check_interval=30):
        while True:
            ft_results = self.retrieve_fine_tuning_results_mistral(finetune_job_id)
            status = ft_results.status
            print(f"Current job status: {status}")

            if status == 'VALIDATED':
                print("Job has been validated. Starting the job...")
                self.client.fine_tuning.jobs.start(job_id=finetune_job_id)
                break

            elif status in ('FAILED', 'FAILED_VALIDATION'):
                print("Fine-tuning job failed.")
                return ft_results

            elif status in ('QUEUED', 'STARTED', 'VALIDATING', 'RUNNING', 'CANCELLED', 'CANCELLATION_REQUESTED'):
                print(f"Waiting for the job to be validated... (status: {status})")
                time.sleep(check_interval)

            else:
                print(f"Unexpected status: {status}.")
                return ft_results

        while True:
            ft_results = self.retrieve_fine_tuning_results_mistral(finetune_job_id)
            status = ft_results.status
            print(f"Current job status: {status}")

            if status == 'SUCCESS':
                data_dict = self.to_dict(ft_results)
                self.save_ft_results(data_dict, model, fold_number)
                print("Fine-tuning job succeeded!")
                return ft_results

            elif status in ('FAILED', 'FAILED_VALIDATION'):
                print("Fine-tuning job failed.")
                return ft_results

            elif status in ('QUEUED', 'STARTED', 'VALIDATING', 'RUNNING', 'CANCELLED', 'CANCELLATION_REQUESTED'):
                print(f"Waiting for the job to finish... (status: {status})")
                time.sleep(check_interval)

            else:
                print(f"Unexpected status: {status}.")
                return ft_results

    def save_ft_results(self, ft_results, model, fold_nr):
        file_name = f"result_file_{model}_{fold_nr}.txt"
        file_path = os.path.join(self.result_file_path, file_name)

        with open(file_path, 'a') as file:
            json.dump(ft_results, file, indent=4)

        print(f"Saved fine-tuning results to {file_path}")

    def fine_tune_model_mistral(self, training_file, validation_file, model, fold_number):
        model_registry = ModelRegistry('fine_tuning_files/mistral/created_models',
                                       "model_registry_mistral")
        upload_response_train_file = self.upload_data_to_mistral_api(training_file)
        print(upload_response_train_file)

        upload_response_validation_file = self.upload_data_to_mistral_api(validation_file)
        print(upload_response_validation_file)

        training_file_id = upload_response_train_file.id
        validation_file_id = upload_response_validation_file.id

        created_job = self.create_model_mistral(training_file_id,
                                                validation_file_id,
                                                model)
        print(created_job)
        fine_tune_results = self.wait_for_job_to_finish(created_job.id, model, fold_number)
        print("fine_tune_results: ", fine_tune_results)
        model_registry.add_model_entry(training_file_id,
                                       validation_file_id,
                                       fine_tune_results.fine_tuned_model,
                                       model)


def construct_file_path(base_directory: Path, path_components: List[str], file_name="") -> str:
    full_path = base_directory.joinpath(*path_components)
    full_path.mkdir(parents=True, exist_ok=True)

    if file_name:
        full_path = full_path / file_name

    return str(full_path)


if __name__ == "__main__":
    base_dir = Path(os.getenv('PROJECT_BASE_DIR'))
    model_name = "open-mistral-7b"
    fold_nr = "fold_0"

    mistral_message_object = MistralMessageObject(model_name)
    mistral_message_object.run()

    fine_tuning_dir_mistral = os.path.join(base_dir,
                                           'fine_tuning_files',
                                           'mistral',
                                           'fine_tuning_data',
                                           'total_ft_data')
    fine_tuning_files_mistral = os.path.join(fine_tuning_dir_mistral, 'file-finetune-mistral.jsonl')
    output_directory_mistral = os.path.join(base_dir,
                                            'fine_tuning_files',
                                            'mistral',
                                            'fine_tuning_data')
    data_splitter = PerformDataSplit(fine_tuning_files_mistral, 5, output_directory_mistral, model_name)
    data_splitter.generate_train_val_sets()

    result_file_dir = construct_file_path(base_dir,
                                          ["fine_tuning_files",
                                           "mistral",
                                           "fine_tuning_evaluation",
                                           f"model_{model_name}",
                                           "metrics"])

    mistral_interaction_for_ft = MistralApiInteractionForFineTuning(result_file_dir)

    train_file = construct_file_path(base_dir,
                                     [
                                         "fine_tuning_files",
                                         "mistral",
                                         "fine_tuning_data",
                                         fold_nr], "train.jsonl")

    val_file = construct_file_path(base_dir,
                                   [
                                       "fine_tuning_files",
                                       "mistral",
                                       "fine_tuning_data",
                                       fold_nr],
                                   "val.jsonl")

    mistral_interaction_for_ft.fine_tune_model_mistral(train_file, val_file, model_name, fold_nr)
