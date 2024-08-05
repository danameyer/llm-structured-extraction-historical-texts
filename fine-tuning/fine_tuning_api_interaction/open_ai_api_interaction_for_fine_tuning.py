import os
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

    def retrieve_fine_tuning_results(self, ft_job_id):
        fine_tune_results = self.client.fine_tuning.jobs.retrieve(ft_job_id)
        return fine_tune_results


if __name__ == "__main__":
    base_dir = Path(os.getenv('PROJECT_BASE_DIR'))
    test_file = os.path.join(base_dir, "test_data", "fine_tuning", "file-finetune.jsonl")
    gpt_model = 'gpt-3.5-turbo'
    open_ai_interaction_for_fine_tuning = OpenAIAPIInteractionForFineTuning()
    # upload_response = open_ai_interaction_for_fine_tuning.upload_data_to_api(test_file)
    # print(upload_response)
#    response_model_creation = open_ai_interaction_for_fine_tuning.create_model('file-JDSPoEc3YHJbbDIkMjPb2Dfr', gpt_model)
#    print(response_model_creation)
    # id='ftjob-f2NsMgumVR1cCzmlLNiZoZMN'
    fine_tune_results = open_ai_interaction_for_fine_tuning.retrieve_fine_tuning_results('ftjob-f2NsMgumVR1cCzmlLNiZoZMN')
    print(fine_tune_results.finished_at)
