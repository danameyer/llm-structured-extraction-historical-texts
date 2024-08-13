import base64
import os
import time
from pathlib import Path
from typing import Optional, List
from dotenv import load_dotenv
from openai import OpenAI
from create_model_registry import ModelRegistry
from fine_tuning.data_split.perform_data_split import PerformDataSplit
from fine_tuning.message_objects.open_ai_message_object import OpenAiMessageObject

load_dotenv()
openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY")


class OpenAIAPIInteractionForFineTuning:
    def __init__(self, result_file_path):
        self.client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
        )
        self.result_file_path = result_file_path
        os.makedirs(result_file_path, exist_ok=True)

    def upload_data_to_api(self, data):
        with open(data, "rb") as file:
            response = self.client.files.create(
                file=file,
                purpose="fine-tune"
            )
        return response

    def create_model(self, train_file, validation_file, model):
        response = self.client.fine_tuning.jobs.create(
            training_file=train_file,
            model=model,
            validation_file=validation_file
        )
        return response

    def retrieve_fine_tuning_results(self, finetune_job_id):
        ft_results = self.client.fine_tuning.jobs.retrieve(finetune_job_id)
        return ft_results

    def get_result_file(self, fine_tuning_job_id, model):
        ft_results = self.retrieve_fine_tuning_results(fine_tuning_job_id)
        result_files = ft_results.result_files

        for result_file in result_files:
            file_name = f"result_file_{model}.txt"
            file_path = os.path.join(self.result_file_path, file_name)
            file = self.client.files.retrieve(result_file)
            content = self.client.files.content(file.id)
            with open(file_path, 'wb') as file:
                file.write(base64.b64decode(content.text.encode("utf-8")))

            print(f"Saved {file_name} to {file_path}")

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

    @staticmethod
    def construct_file_path(base_directory: Path, path_components: List[str], file_name="") -> str:
        full_path = base_directory.joinpath(*path_components)
        full_path.mkdir(parents=True, exist_ok=True)

        if file_name:
            full_path = full_path / file_name

        return str(full_path)

    @staticmethod
    def fine_tune_model(training_file, validation_file, result_file_directory, gpt_model_path):
        open_ai_interaction = OpenAIAPIInteractionForFineTuning(result_file_directory)
        model_registry = ModelRegistry('fine_tuning_files/openai/created_models',
                                       "model_registry_openai")

        upload_response_train_file = open_ai_interaction.upload_data_to_api(training_file)
        print(upload_response_train_file)

        upload_response_validation_file = open_ai_interaction.upload_data_to_api(validation_file)
        print(upload_response_validation_file)

        training_file_id = upload_response_train_file.id
        existing_model_id = model_registry.get_model_id(training_file_id)

        validation_file_id = upload_response_validation_file.id

        if existing_model_id:
            print(f"Using existing model: {existing_model_id}")
        else:
            response_model_creation = open_ai_interaction.create_model(train_file=training_file_id,
                                                                       model=gpt_model_path,
                                                                       validation_file=validation_file_id)
            print(response_model_creation)

            ft_job_id = response_model_creation.id
            fine_tune_results = open_ai_interaction.wait_for_job_to_finish(ft_job_id)
            print(fine_tune_results.finished_at)

            model_registry.add_model_id(training_file_id, validation_file_id, ft_job_id, gpt_model_path)
            open_ai_interaction.get_result_file(ft_job_id, gpt_model_path)

        # model_registry.add_model_id("file-bUlvAiXzLJCY5YEGBBmboEEh",
        #                             "file-bUlvAiXzLJCY5YEGBBmboEEh",
        #                             "ftjob-DAxstQl595n0nHwNfA3xUelp",
        #                             gpt_model_path)
        # open_ai_interaction.get_result_file(fine_tuning_job_id="ftjob-DAxstQl595n0nHwNfA3xUelp",
        #                                     model=gpt_model_path)


if __name__ == "__main__":
    base_dir = Path(os.getenv('PROJECT_BASE_DIR'))
    gpt_model = 'gpt-3.5-turbo'
    open_ai_message_object = OpenAiMessageObject("gpt-3.5-turbo")
    open_ai_message_object.run()

    fine_tuning_dir = os.path.join(base_dir,
                                   'fine_tuning_files',
                                   "openai",
                                   'fine_tuning_data',
                                   'total_ft_data')
    fine_tuning_files = os.path.join(fine_tuning_dir, 'file-finetune-openai.jsonl')
    output_directory = os.path.join(base_dir,
                                    'fine_tuning_files',
                                    "openai",
                                    'fine_tuning_data')
    data_splitter = PerformDataSplit(fine_tuning_files, 5, output_directory, gpt_model)
    data_splitter.generate_train_val_sets()

    open_ai_interaction_for_ft = OpenAIAPIInteractionForFineTuning
    train_file = open_ai_interaction_for_ft.construct_file_path(base_dir,
                                                                [
                                                                    "fine_tuning_files",
                                                                    "openai",
                                                                    "fine_tuning_data",
                                                                    "fold_0"], "train.jsonl")

    val_file = open_ai_interaction_for_ft.construct_file_path(base_dir,
                                                              [
                                                                  "fine_tuning_files",
                                                                  "openai",
                                                                  "fine_tuning_data",
                                                                  "fold_0"],
                                                              "val.jsonl")

    result_file_dir = open_ai_interaction_for_ft.construct_file_path(base_dir,
                                                                     ["fine_tuning_files",
                                                                      "openai",
                                                                      "fine_tuning_evaluation",
                                                                      f"model_{gpt_model}",
                                                                      "metrics"])

    open_ai_interaction_for_ft.fine_tune_model(train_file, val_file, result_file_dir, gpt_model)

