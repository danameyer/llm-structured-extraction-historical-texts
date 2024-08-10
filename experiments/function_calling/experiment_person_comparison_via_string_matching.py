from function_calling_components.chat_completion_blueprint import DialogueCompletion
from functions.PersonComparisonViaStringMatching import DecideIfSamePerson


class ExperimentPersonComparisonViaStringMatching:
    def __init__(self, experiment_dir):
        self.experiment_dir = experiment_dir
        self.dialogue = DialogueCompletion(model='gpt-3.5-turbo', experiment_dir=experiment_dir)

    def generate_prompt(self):
        prompt = "Decide if two people of the same name are the same. The name of the person is Willelmus."
        prompt_name = "person_comparison_via_string_matching"
        return prompt, prompt_name

    def run(self):
        prompt, filename = self.generate_prompt()
        decide_if_same_person = DecideIfSamePerson()
        function_list = [decide_if_same_person]
        self.dialogue.prompt_assistant_response(prompt, filename, function_list, validate=False)


if __name__ == '__main__':
    experiment_1 = ExperimentPersonComparisonViaStringMatching("/tmp/experiments/comparison_via_string_matching")
    experiment_1.run()
