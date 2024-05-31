import json
from pathlib import Path
from typing import List, Dict, Optional, Union

import jsons
from dotenv import load_dotenv
import os
import openai
from openai import OpenAI
from openai.types import Completion
from openai.types.chat import ChatCompletion

from function_calling_components.function_calling import Message
from functions.BaseFunction import BaseFunction
from datetime import datetime

load_dotenv()
openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
selected_model: str = "gpt-3.5-turbo-0125"


class DialogueCompletion:

    def __init__(self, model: str):
        load_dotenv()
        self.client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
        )
        self.model: str = model
        self.message_history: List[Message] = []
        base_dir = Path(os.getenv('PROJECT_BASE_DIR'))
        self.prompt_save_dir = os.path.join(base_dir, "prompting", "generated_prompts")
        self.response_save_dir = os.path.join(base_dir, "prompting", "prompting_responses")
        os.makedirs(self.prompt_save_dir, exist_ok=True)
        os.makedirs(self.response_save_dir, exist_ok=True)
        self.current_prompt_filename = None

    def request_response(self,
                         messages: List[Message],
                         functions: List[BaseFunction] = None,
                         function_call="auto") -> Union[ChatCompletion, None]:
        """
        Generate response using OpenAI Chat Completion API.

        Args:
            self: self parameter
            messages (list): List of message objects.
            functions (list, optional): List of function dictionaries. Defaults to None.
            function_call (str, optional): Type of function call to use. Defaults to "auto".

        Returns:
            Completion: Generated response.
        """
        try:
            messages_as_dict = json.loads(jsons.dumps(messages))
            functions_as_dict_list = [x.get_definition_dict() for x in functions] if functions else None

            completion: Union[ChatCompletion, None] = self.client.chat.completions.create(
                model=self.model,
                messages=messages_as_dict,
                functions=functions_as_dict_list,
                function_call=function_call if functions else None
            )

            # function definitions have been sent to chatGPT - now only use the short description in order to
            # save tokens ...
            self.flag_function_calls_for_short_description(functions)

            return completion
        except openai.APIConnectionError as e:
            print("The server could not be reached")
            print(e.__cause__)
        except openai.RateLimitError as e:
            print("Rate limit has been exceeded.")
            print(f"Exception: {e}")
        except openai.APIStatusError as e:
            print("Another non-200-range status code was received")
            print(e.status_code)
            print(e.response)
        except openai.OpenAIError as e:
            print("An unexpected API error occurred.")
            print(f"Exception: {e}")

    def append_message(self, message: Message):
        self.message_history.append(message)

    def print_conversation(self):
        role_to_color: Dict[str, str] = {
            "system": "\033[96m",  # Cyan
            "user": "\033[93m",  # Yellow
            "assistant": "\033[95m",  # Purple
            "function": "\033[97m",  # White
        }
        reset_color: str = "\033[0m"  # Reset color to default

        for message in self.message_history:
            role: str = message.role
            content: str = message.content
            colored_content: str = f"{role_to_color[role]}{content}{reset_color}"
            print(f"{role}: {colored_content}\n\n")

    def execute_chat_completion_query(self,
                                      messages: List[Message],
                                      functions: List[BaseFunction] = None) -> Union[ChatCompletion, None]:
        completion = self.request_response(messages, functions)
        message_output = completion.choices[0]

        if message_output.finish_reason == "function_call":
            print("Function will be called.")
            function_call_result = self.perform_function_call(completion, messages, functions)
            return function_call_result
        else:
            print("No function called.")
            return completion

    def perform_function_call(self,
                              completion: Union[ChatCompletion, None],
                              messages: List[Message],
                              functions: List[BaseFunction]) -> Union[ChatCompletion, None]:
        function_name = completion.choices[0].message.function_call.name
        function_parameters = json.loads(
            completion.choices[0].message.function_call.arguments)

        function_object = next((x for x in functions if x.get_definition().name == function_name), None)

        if function_object:
            try:
                output = function_object.run(**function_parameters)
            except Exception as e:
                print(function_parameters)
                print(f"Function could not be called.")
                print(f"Error message: {e}")
                output = f"Function could not be called. error: {str(e)}"
            messages.append(
                Message(role="function",
                        content=str(output),
                        name=completion.choices[0].message.function_call.name)
            )
            try:
                response = self.request_response(messages)
                return response
            except Exception as e:
                print(type(e))
                raise Exception("Chat response could not be generated.")

    def save_to_file(self, out_path, content, filename):
        file_path = os.path.join(out_path, filename)
        with open(file_path, 'a') as file:
            file.write(content + '\n')

    def prompt_user_input(self):
        user_input = input("You: ")
        self.append_message(Message(role="user", content=user_input))
        response = self.execute_chat_completion_query(self.message_history)
        if response:
            assistant_message = response.choices[0].message.content
            self.append_message(Message(role="assistant", content=assistant_message))
            self.print_conversation()

            if self.current_prompt_filename:
                self.save_to_file(self.response_save_dir, f"User: {user_input}", self.current_prompt_filename)
                self.save_to_file(self.response_save_dir, f"Assistant: {assistant_message}", self.current_prompt_filename)

    def perform_initial_prompting(self, prompt_choice, function_list=None):
        prompt, prompt_name = prompt_choice()
        self.append_message(Message("user", prompt))

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.current_prompt_filename = f"{prompt_name}_{timestamp}.txt"

        self.save_to_file(self.prompt_save_dir, prompt, self.current_prompt_filename)

        chat_response = self.execute_chat_completion_query(
            messages=self.message_history,
            functions=function_list
        )
        assistant_message = chat_response.choices[0].message.content

        self.append_message(Message("assistant", assistant_message))
        self.save_to_file(self.response_save_dir, assistant_message, self.current_prompt_filename)
        self.print_conversation()

    def flag_function_calls_for_short_description(self, functions: Union[List[BaseFunction], None]):
        if functions:
            for f in functions:
                f.flag_use_short_definition_true()
