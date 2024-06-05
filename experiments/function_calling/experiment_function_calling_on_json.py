from function_calling_components.chat_completion_blueprint import DialogueCompletion
from functions.ExtractPersonInfo import ExtractPersonInfo


class Experiment_Function_Calling_on_JSON:
    def __init__(self):
        self.dialogue = DialogueCompletion(model='gpt-3.5-turbo')

    def generate_prompt(self):
        prompt = "Please extract all the information about every person from the JSON file. Ask for clarification if you don't know the name. The name of the person is Willelmus:"
        prompt_name = "extract_person_info_from_json"
        return prompt, prompt_name

    def run(self):
        prompt, filename = self.generate_prompt()
        extract_person_info = ExtractPersonInfo()
        function_list = [extract_person_info]
        self.dialogue.prompt_assistant_response(prompt, filename, function_list)


if __name__ == '__main__':
    experiment_1 = Experiment_Function_Calling_on_JSON()
    experiment_1.run()
    