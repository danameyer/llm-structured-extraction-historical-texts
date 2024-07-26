import json
from pathlib import Path
from typing import List, Dict, Optional, Union

import jsons
from dotenv import load_dotenv
import os
import openai
from jsons import ValidationError
from openai import OpenAI
from openai.types import Completion
from openai.types.chat import ChatCompletion

from evaluation.json_comparison.json_validation import JsonValidator
from function_calling_components.chat_file_writer import ChatFileWriter
from function_calling_components.function_calling import Message, SimpleChatGptMessage
from function_calling_components.token_counting import TokenCounter
from functions.BaseFunction import BaseFunction
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)

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
        self.chat_file_writer = ChatFileWriter()
        self.function_call_result = None
        self.token_counter = TokenCounter(self.model)

    def _request_response(self,
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
            simple_chat_gpt_messages = [SimpleChatGptMessage(role=x.role, content=x.content, name=x.name) for x in messages]
            messages_as_dict = json.loads(jsons.dumps(simple_chat_gpt_messages))
            print(f"Input response content: {messages_as_dict}")
            functions_as_dict_list = [x.get_definition_dict() for x in functions] if functions else None

            completion: Union[ChatCompletion, None] = self.client.chat.completions.create(
                model=self.model,
                messages=messages_as_dict,
                functions=functions_as_dict_list,
                function_call=function_call if functions else None
            )

            output_response = completion.choices[0].message.content
            print(f"Output response content: {output_response}")

            # self.token_counter.add_input(messages_as_dict)
            # self.token_counter.add_output(output_response)
            self.token_counter.add_number_of_input_tokens(completion.usage.prompt_tokens)
            self.token_counter.add_number_of_output_tokens(completion.usage.completion_tokens)

            # function definitions have been sent to chatGPT - now only use the short description in order to
            # save tokens ...
            # self.flag_function_calls_for_short_description(functions)
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

    def _append_message(self, message: Message):
        self.message_history.append(message)

    def _print_conversation(self):
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

    @retry(stop=stop_after_attempt(6), wait=wait_random_exponential(multiplier=1, max=10), reraise=True,)
    def _execute_chat_completion_query(self,
                                       messages: List[Message],
                                       functions: List[BaseFunction] = None,
                                       validate=True) -> Union[ChatCompletion, None]:
        completion = self._request_response(messages, functions)
        message_output = completion.choices[0]

        if message_output.finish_reason == "function_call":
            print("Function will be called.")
            function_call_result = self._perform_function_call(completion, messages, functions, validate)
            return function_call_result
        else:
            print("No function called.")
            return completion

    def _perform_function_call(self,
                               completion: Union[ChatCompletion, None],
                               messages: List[Message],
                               functions: List[BaseFunction],
                               validate=True) -> Union[ChatCompletion, None]:
        function_name = completion.choices[0].message.function_call.name
        function_parameters = json.loads(
            completion.choices[0].message.function_call.arguments)
        print("These are the function call arguments:", function_parameters)

        function_object = next((x for x in functions if x.get_definition().name == function_name), None)

        if function_object:
            # try:
            self.function_call_result = function_object.run(**function_parameters)
            print("This is the function call result", self.function_call_result)
            json_validator = JsonValidator()

            if validate:
                validation = json_validator.validate_json(self.function_call_result)
                if not validation[0]:
                    message = "JSON validation failed: " + validation[1]
                    raise ValidationError(message)

            messages.append(
                Message(role="function",
                        content=str(self.function_call_result),
                        name=completion.choices[0].message.function_call.name,
                        # function_call_id=completion.choices[0].message.tool_calls[0].id,
                        function_call_arguments=completion.choices[0].message.function_call.arguments
                        )
            )
            try:
                response = self._request_response(messages)
                return response
            except Exception as e:
                print(type(e))
                raise Exception("Chat response could not be generated.")

    def add_dynamic_prompting(self, filename, print_conversation=True):
        user_input = input("You: ")
        self._append_message(Message(role="user", content=user_input))
        response = self._execute_chat_completion_query(self.message_history)
        if response:
            assistant_message = response.choices[0].message.content
            self._append_message(Message(role="assistant", content=assistant_message))
            if print_conversation:
                self._print_conversation()

            self.chat_file_writer.save_response(f"User: {user_input}", filename)
            self.chat_file_writer.save_response(f"Assistant: {assistant_message}", filename)

    def prompt_assistant_response(self,
                                  prompt,
                                  filename,
                                  function_list=None,
                                  print_conversation=True,
                                  validate=True):
        self._append_message(Message("user", prompt))

        self.chat_file_writer.save_prompt(prompt, filename)

        chat_response = self._execute_chat_completion_query(
            messages=self.message_history,
            functions=function_list,
            validate=validate
        )
        assistant_message = chat_response.choices[0].message.content

        self._append_message(Message("assistant", assistant_message))
        self.chat_file_writer.save_response(assistant_message, filename)

        if print_conversation:
            self._print_conversation()

        return assistant_message

    def add_system_prompt(self, prompt, filename, print_conversation=True):
        self._append_message(Message("system", prompt))

        self.chat_file_writer.save_prompt(prompt, filename)

        if print_conversation:
            self._print_conversation()

    def flag_function_calls_for_short_description(self, functions: Union[List[BaseFunction], None]):
        if functions:
            for f in functions:
                f.flag_use_short_definition_true()
