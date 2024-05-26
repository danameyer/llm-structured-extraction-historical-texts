from function_calling_components.chat_completion_blueprint import DialogueCompletion
from function_calling_components.function_calling import Message
from functions.ExtractPersonInfo import ExtractPersonInfo


class Experiment1:
    def __init__(self):
        pass

    def run(self):
        extract_person_info = ExtractPersonInfo()

        dialogue = DialogueCompletion(model='gpt-3.5-turbo')
        dialogue.append_message(Message("user",
                                        "Please extract all the information about every person from the JSON file. Ask for clarification if you don't know the name. The name of the person is Willelmus:"))

        function_list = [extract_person_info]

        chat_response = dialogue.execute_chat_completion_query(
            messages=dialogue.message_history,
            functions=function_list
        )
        assistant_message = chat_response.choices[0].message.content
        dialogue.append_message(Message("assistant", assistant_message))
        dialogue.print_conversation()


if __name__ == '__main__':
    experiment_1 = Experiment1()
    experiment_1.run()
    