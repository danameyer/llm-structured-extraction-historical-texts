from function_calling_components.chat_completion_blueprint import DialogueCompletion
from function_calling_components.function_calling import Message
from functions.ExtractPersonInfo import ExtractPersonInfo


class Experiment1:
    def __init__(self):
        self.dialogue = DialogueCompletion(model='gpt-3.5-turbo')

    def generate_prompt(self):
        prompt = "Please extract all the information about every person from the JSON file. Ask for clarification if you don't know the name. The name of the person is Willelmus:"
        prompt_name = "extract_person_info_from_json"
        return prompt, prompt_name

    def run_experiment(self, prompt_choice):
        extract_person_info = ExtractPersonInfo()
        function_list = [extract_person_info]
        self.dialogue.prompt_assistant_response(prompt_choice, filename, function_list)

    def run(self):
        self.run_experiment(self.generate_prompt)


if __name__ == '__main__':
    experiment_1 = Experiment1()
    experiment_1.run()
    