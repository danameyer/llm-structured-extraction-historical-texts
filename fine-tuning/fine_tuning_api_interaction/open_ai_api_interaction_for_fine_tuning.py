from openai import OpenAI

client = OpenAI()


class OpenAIAPIInteractionForFineTuning:
    def __init__(self):
        pass

    def upload_data_to_api(self, data):
        client.files.create(
            file=open(data, "rb"),
            purpose="fine-tune"
        )

    def create_model(self, file, model):
        client.fine_tuning.jobs.create(
            training_file=file,
            model=model
        )
