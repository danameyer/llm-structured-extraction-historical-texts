import os
from pathlib import Path

from dotenv import load_dotenv

from function_calling_components.chat_completion_blueprint import DialogueCompletion
from function_calling_components.function_calling import Message
from prompting.prompting_strategies import PromptBuilder


class Experiment2:

    def __init__(self):
        load_dotenv()

    def run(self):
        base_dir = Path(os.getenv('PROJECT_BASE_DIR'))
        print(base_dir)
        test_file_path = os.path.join(base_dir, "test_data", "test_txt", "sample_text.txt")
        print(test_file_path)
        test_files = [test_file_path]
        prompt_builder = PromptBuilder(test_files)
        prompt = prompt_builder.prompting_strategies

        dialogue = DialogueCompletion(model='gpt-3.5-turbo')
        dialogue.append_message(Message("user", prompt))

        chat_response = dialogue.execute_chat_completion_query(
            messages=dialogue.message_history,
        )
        assistant_message = chat_response.choices[0].message.content
        dialogue.append_message(Message("assistant", assistant_message))
        dialogue.print_conversation()


if __name__ == '__main__':
    experiment_2 = Experiment2()
    experiment_2.run()
