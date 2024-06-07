from function_calling_components.chat_completion_blueprint import DialogueCompletion
from functions.PersonComparisonViaStringMatching import DecideIfSamePerson


class ExperimentPersonComparisonViaStringMatching():
    def __init__(self):
        self.dialogue = DialogueCompletion(model='gpt-3.5-turbo')

    def generate_prompt(self):
        prompt = "Decide if two people of the same name are the same. The name of the person is Willelmus."
        prompt_name = "extract_person_info_from_json"
        return prompt, prompt_name

    def run(self):
        prompt, filename = self.generate_prompt()
        decide_if_same_person = DecideIfSamePerson()
        function_list = [decide_if_same_person]
        self.dialogue.prompt_assistant_response(prompt, filename, function_list)


if __name__ == '__main__':
    experiment_1 = ExperimentPersonComparisonViaStringMatching()
    experiment_1.run()
