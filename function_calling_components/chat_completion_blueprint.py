import json
from typing import List, Dict, Optional, Union

import jsons
from dotenv import load_dotenv
import os
import openai
from openai import OpenAI
from openai.types import Completion
from openai.types.chat import ChatCompletion

from function_calling_components.function_calling import Message
from functions.ExtractPersonInfo import ExtractPersonInfo

# from functions.ExtractPersonInfo import ExtractPersonInfo

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

    def request_response(self,
                         messages: List[Message],
                         functions: Optional[List[Dict[str, Union[str, List[str]]]]] = None,
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

            completion: Union[ChatCompletion, None] = self.client.chat.completions.create(
                model=self.model,
                messages=messages_as_dict,
                functions=functions,
                function_call=function_call if functions else None
            )
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
                                      functions: Optional[List[Dict[str, Union[str, List[str]]]]] = None) -> Union[ChatCompletion, None]:
        completion = self.request_response(messages, functions)
        message_output = completion.choices[0]

        if message_output.finish_reason == "function_call":
            print("Function will be called.")
            function_call_result = self.perform_function_call(completion, messages)
            return function_call_result
        else:
            print("No function called.")
            return completion

    def perform_function_call(self, completion: Union[ChatCompletion, None], messages: List[Message]) -> Union[
        ChatCompletion, None]:
        # json_info_extractor = JsonInfoExtractor()
        # globals()["extract_person_info"] = json_info_extractor.extract_person_info
        if completion.choices[0].message.function_call.name == "extract_person_info":
            try:
                parameters: Dict[str, Union[str, None]] = json.loads(
                    completion.choices[0].message.function_call.arguments
                )
                name: Union[str, None] = parameters.get("name")
                output = ExtractPersonInfo.run(name)
            except Exception as e:
                print(parameters)
                print(f"Function could not be called.")
                print(f"Error message: {e}")
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
