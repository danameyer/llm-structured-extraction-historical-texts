import base64
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List
from dotenv import load_dotenv
from openai import OpenAI
from create_model_registry import ModelRegistry
from fine_tuning.data_split.perform_data_split import PerformDataSplit
from fine_tuning.message_objects.open_ai_message_object import OpenAiMessageObject
from function_calling_components.model_pricing import ModelPricing

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

    def get_result_file(self, fine_tuning_job_id, model, fold_nr):
        ft_results = self.retrieve_fine_tuning_results(fine_tuning_job_id)
        result_files = ft_results.result_files

        for result_file in result_files:
            file_name = f"result_file_{model}_{fold_nr}.txt"
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
    def fine_tune_model(training_file, validation_file, result_file_directory, gpt_model_path, fold_nr):
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
            fine_tuned_model = fine_tune_results.fine_tuned_model
            print(fine_tuned_model)
            epoch_num = fine_tune_results.hyperparameters.n_epochs
            token_num = fine_tune_results.trained_tokens
            ft_start = fine_tune_results.created_at
            ft_end = fine_tune_results.finished_at
            # Convert Unix timestamps to datetime objects
            created_datetime = datetime.fromtimestamp(ft_start, timezone.utc)
            finished_datetime = datetime.fromtimestamp(ft_end, timezone.utc)

            # Calculate the duration
            duration = finished_datetime - created_datetime
            duration_as_seconds = duration.total_seconds()

            model_pricing = ModelPricing()
            pricing_per_token = model_pricing.get_price_per_token(model_name=gpt_model_path,
                                                                  is_finetune=True)
            training_price = pricing_per_token["training"] * token_num * epoch_num

            model_registry.add_model_entry(training_file_id,
                                           validation_file_id,
                                           ft_job_id,
                                           fine_tuned_model,
                                           gpt_model_path,
                                           fold_nr,
                                           token_num,
                                           epoch_num,
                                           duration_as_seconds,
                                           training_price)
            open_ai_interaction.get_result_file(ft_job_id, gpt_model_path, fold_nr)

        # test scenario

        # ft_job_id = "ftjob-8u0HVcsAwMtmZY7ueUHNi2Qw"
        # training_file_id = "file-oaGESJuM0Bj9k3CajeDCHoNw"
        # validation_file_id = "file-4Ax1TecpCFcEvxw9DixDVmqf"
        # fine_tuned_model = "ft:gpt-4o:personal::9sv2TPIE"
        # epoch_num = 5
        # token_num = 5768
        # created_datetime = datetime.fromtimestamp(1692661014, timezone.utc)
        # finished_datetime = datetime.fromtimestamp(1692661190, timezone.utc)
        # duration = finished_datetime - created_datetime
        # duration_as_seconds = duration.total_seconds()
        #
        # model_pricing = ModelPricing()
        # pricing_per_token = model_pricing.get_price_per_token(model_name=gpt_model_path, is_finetune=True)
        # training_price = pricing_per_token["training"] * token_num * epoch_num
        #
        # model_registry.add_model_entry(training_file_id,
        #                                validation_file_id,
        #                                ft_job_id,
        #                                fine_tuned_model,
        #                                gpt_model_path,
        #                                fold_nr,
        #                                token_num,
        #                                epoch_num,
        #                                duration_as_seconds,
        #                                training_price)
        # open_ai_interaction.get_result_file(ft_job_id, gpt_model_path, fold_nr)


if __name__ == "__main__":
    base_dir = Path(os.getenv('PROJECT_BASE_DIR'))
    gpt_model = 'gpt-4o-mini-2024-07-18'
    fold_nr = 'fold_0'

    # Data splitting
    fine_tuning_dir = os.path.join(base_dir,
                                   'fine_tuning_files',
                                   "openai",
                                   'fine_tuning_data',
                                   'total_ft_data')
    fine_tuning_files = os.path.join(fine_tuning_dir,
                                     'file-finetune-openai.jsonl')
    output_directory_data_split = os.path.join(base_dir,
                                               'fine_tuning_files',
                                               "openai",
                                               'fine_tuning_data')
    output_directory_fine_tuning_evaluation_data = os.path.join(base_dir,
                                                                'evaluation_results',
                                                                'txt_files_fine_tuning_evaluation_data')
    path_to_text_files = os.path.join(base_dir,
                                      'data',
                                      'test_txt')

    # Only perform data splitting if no old runs exist
    try:
        open_ai_message_object = OpenAiMessageObject(gpt_model)
        open_ai_message_object.run()

        data_splitter = PerformDataSplit(fine_tuning_files, 3, output_directory_data_split, gpt_model)
        fold_info_list = data_splitter.generate_train_val_sets()

        data_splitter.copy_text_files(fold_info_list=fold_info_list,
                                      path_to_text_files=path_to_text_files,
                                      output_directory=output_directory_fine_tuning_evaluation_data)

    except FileExistsError as file_exists_error:
        print(f"""
                One operation failed:
                  - creating OpenAiMessageObject or 
                  - splitting data or 
                  - copying text files for fine-tuning evaluation
                
                This error is raised because old files are still present.
                Some steps will be skipped and the pipeline will continue with the actual fine-tuning.
                
                Details: {file_exists_error}
                """)

    open_ai_interaction_for_ft = OpenAIAPIInteractionForFineTuning
    train_file = open_ai_interaction_for_ft.construct_file_path(base_dir,
                                                                [
                                                                    "fine_tuning_files",
                                                                    "openai",
                                                                    "fine_tuning_data",
                                                                    fold_nr], "train.jsonl")

    val_file = open_ai_interaction_for_ft.construct_file_path(base_dir,
                                                              [
                                                                  "fine_tuning_files",
                                                                  "openai",
                                                                  "fine_tuning_data",
                                                                  fold_nr],
                                                              "val.jsonl")

    result_file_dir = open_ai_interaction_for_ft.construct_file_path(base_dir,
                                                                     ["fine_tuning_files",
                                                                      "openai",
                                                                      "fine_tuning_evaluation",
                                                                      f"model_{gpt_model}",
                                                                      "metrics"])

    # open_ai_interaction_for_ft.fine_tune_model(train_file, val_file, result_file_dir, gpt_model, fold_nr)
