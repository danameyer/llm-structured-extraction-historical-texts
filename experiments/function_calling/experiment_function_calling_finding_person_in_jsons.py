from function_calling_components.chat_completion_blueprint import DialogueCompletion
from functions.FindPersonInJsonFiles import PersonFinder


class ExperimentFunctionCallingFindingPersonInJsons:
    def __init__(self):
        self.dialogue = DialogueCompletion(model='gpt-3.5-turbo')

    def generate_prompt(self):
        prompt = "I want to find the 5 most similar persons to the person with the name Simon and the cognomen Leukenor. Please return the results in JSON format, including all available attributes for each person."
        prompt_name = "find_person_in_json_files"
        return prompt, prompt_name

    def run(self):
        prompt, filename = self.generate_prompt()
        find_person_in_json_files = PersonFinder()
        function_list = [find_person_in_json_files]
        self.dialogue.prompt_assistant_response(prompt, filename, function_list, validate=False)
        function_call_result = self.dialogue.function_call_result
        return function_call_result


if __name__ == '__main__':
    experiment = ExperimentFunctionCallingFindingPersonInJsons()
    response_json = experiment.run()
    print("This is the response from the experiment:", response_json)
