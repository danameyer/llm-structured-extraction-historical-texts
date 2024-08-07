import json
import os
import time
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY")


class OpenAIAPIInteractionForFineTuning:
    def __init__(self):
        self.client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
        )

    def upload_data_to_api(self, data):
        with open(data, "rb") as file:
            response = self.client.files.create(
                file=file,
                purpose="fine-tune"
            )
        return response

    def create_model(self, file, model):
        response = self.client.fine_tuning.jobs.create(
            training_file=file,
            model=model
        )
        return response

    def retrieve_fine_tuning_results(self, finetune_job_id):
        ft_results = self.client.fine_tuning.jobs.retrieve(finetune_job_id)
        return ft_results

    def wait_for_job_to_finish(self, finetune_job_id, check_interval=30):
        while True:
            ft_results = self.retrieve_fine_tuning_results(finetune_job_id)
            status = ft_results.status
            print(f"Current job status: {status}")
            if status == 'succeeded':
                print("Fine-tuning job succeeded!")
                return ft_results
            elif status == 'failed':
                print("Fine-tuning job failed.")
                return ft_results
            else:
                print(f"Waiting for the job to finish... (status: {status})")
                time.sleep(check_interval)


class ModelRegistry:
    def __init__(self, registry_file_name='model_registry.json'):
        base_dir = os.getenv('PROJECT_BASE_DIR', '/default/path')
        if not base_dir:
            raise ValueError("PROJECT_BASE_DIR environment variable not set.")

        # Define the relative path and combine with the base directory
        relative_path = 'test_data/fine_tuning'
        full_path = os.path.join(base_dir, relative_path, registry_file_name)

        # Initialize attributes
        self.registry_file = full_path
        self.registry = None
        self.load_registry()

    def load_registry(self):
        if os.path.exists(self.registry_file):
            with open(self.registry_file, 'r') as file:
                self.registry = json.load(file)
        else:
            self.registry = {}

    def save_registry(self):
        with open(self.registry_file, 'w') as file:
            json.dump(self.registry, file, indent=4)

    def get_model_id(self, train_file_id):
        return self.registry.get(train_file_id, None)

    def add_model_id(self, train_file_id, model_id):
        self.registry[train_file_id] = model_id
        self.save_registry()


if __name__ == "__main__":
    base_dir = Path(os.getenv('PROJECT_BASE_DIR'))
    test_file = os.path.join(base_dir, "test_data", "fine_tuning", "file-finetune.jsonl")
    gpt_model = 'gpt-3.5-turbo'

    open_ai_interaction_for_fine_tuning = OpenAIAPIInteractionForFineTuning()
    model_registry = ModelRegistry()

    # Upload the data to the API
    upload_response = open_ai_interaction_for_fine_tuning.upload_data_to_api(test_file)
    print(upload_response)

    # Extract the training file ID from the upload response
    training_file_id = upload_response.id

    # Check if we already have a model for this training file
    existing_model_id = model_registry.get_model_id(training_file_id)
    if existing_model_id:
        print(f"Using existing model: {existing_model_id}")
    else:
        # Create the fine-tuning model
        response_model_creation = open_ai_interaction_for_fine_tuning.create_model(training_file_id, gpt_model)
        print(response_model_creation)

        # Extract the fine-tuning job ID from the response
        ft_job_id = response_model_creation.id

        # Wait for the fine-tuning job to finish
        fine_tune_results = open_ai_interaction_for_fine_tuning.wait_for_job_to_finish(ft_job_id)
        print(fine_tune_results.finished_at)

        # Record the new model ID
        model_registry.add_model_id(training_file_id, ft_job_id)