import json
import os
from typing import Dict, List, Union, Optional
import openai
from dotenv import load_dotenv
from jsons import ValidationError
from openai import OpenAI
from openai.types.responses import Response
from evaluation.json_comparison.json_validation import JsonValidator
from function_calling_components.chat_file_writer import ChatFileWriter
from function_calling_components.function_calling import Message
from function_calling_components.runtime_calculation import RuntimeCalculation
from function_calling_components.token_counting import TokenCounter
from functions.BaseFunction import BaseFunction
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)

load_dotenv()


class DialogueCompletion:

    def __init__(self, model: str, experiment_dir):
        load_dotenv()
        self.client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
        )
        self.model: str = model
        self.message_history: List[Message] = []
        # base_dir = Path(os.getenv('PROJECT_BASE_DIR'))
        # self.prompt_save_dir = os.path.join(base_dir, "prompting", "generated_prompts")
        # self.response_save_dir = os.path.join(base_dir, "prompting", "prompting_responses")
        # os.makedirs(self.prompt_save_dir, exist_ok=True)
        # os.makedirs(self.response_save_dir, exist_ok=True)
        self.chat_file_writer = ChatFileWriter(experiment_dir)
        self.function_call_result = None
        self.token_counter = TokenCounter(self.model)
        self.runtime_calculator = RuntimeCalculation()

    def _request_response(
            self,
            messages: List[Message],
            functions: Optional[List[BaseFunction]] = None,
            function_call="auto",
    ) -> Union[Response, None]:

        try:
            input_messages = self._messages_to_input(messages)

            print(f"Input response content: {input_messages}")

            self.runtime_calculator.start()

            if functions:
                tools = [
                    self._get_responses_tool_definition(function)
                    for function in functions
                ]

                response = self.client.responses.create(
                    model=self.model,
                    input=input_messages,
                    tools=tools,
                    tool_choice=function_call,
                    parallel_tool_calls=False,
                )
            else:
                response = self.client.responses.create(
                    model=self.model,
                    input=input_messages,
                )

            self.runtime_calculator.end()

            print(
                f"API call duration: "
                f"{self.runtime_calculator.calculate_runtime():.2f} seconds"
            )

            print(f"Output response content: {response.output_text}")

            if response.usage:
                self.token_counter.add_number_of_input_tokens(
                    response.usage.input_tokens
                )
                self.token_counter.add_number_of_output_tokens(
                    response.usage.output_tokens
                )

            return response

        except openai.APIConnectionError as e:
            print("The server could not be reached")
            print(e.__cause__)
            raise

        except openai.RateLimitError as e:
            print("Rate limit has been exceeded.")
            print(f"Exception: {e}")
            raise

        except openai.APIStatusError as e:
            print("Another non-200-range status code was received")
            print(e.status_code)
            print(e.response)
            raise

        except openai.OpenAIError as e:
            print("An unexpected API error occurred.")
            print(f"Exception: {e}")
            raise

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

    @retry(
        stop=stop_after_attempt(6),
        wait=wait_random_exponential(multiplier=1, max=10),
        reraise=True,
    )
    def _execute_chat_completion_query(
            self,
            messages: List[Message],
            tools: Optional[List[BaseFunction]]= None,
            validate=True,
    ) -> Union[Response, None]:

        response = self._request_response(
            messages=messages,
            functions=tools,
        )

        function_calls = [
            item
            for item in response.output
            if item.type == "function_call"
        ]

        if function_calls:
            print("Function will be called.")

            return self._perform_function_call(
                response=response,
                function_call=function_calls[0],
                tools=tools,
                validate=validate,
            )

        print("No function called.")
        return response

    def _perform_function_call(
            self,
            response: Response,
            function_call,
            tools: List[BaseFunction],
            validate=True,
    ) -> Union[Response, None]:

        function_name = function_call.name
        function_parameters = json.loads(function_call.arguments)

        print(
            "These are the function call arguments:",
            function_parameters
        )

        function_object = next(
            (
                function
                for function in tools
                if function.get_definition().function.name == function_name
            ),
            None,
        )

        if not function_object:
            raise ValueError(
                f"Unknown function requested: {function_name}"
            )

        self.function_call_result = function_object.run(
            **function_parameters
        )

        print(
            "This is the function call result",
            self.function_call_result
        )

        if validate:
            json_validator = JsonValidator()
            validation = json_validator.validate_json(
                self.function_call_result
            )

            if not validation[0]:
                message = (
                        "JSON validation failed: "
                        + validation[1]
                )
                raise ValidationError(message)

        try:
            self.runtime_calculator.start()

            final_response = self.client.responses.create(
                model=self.model,
                previous_response_id=response.id,
                input=[
                    {
                        "type": "function_call_output",
                        "call_id": function_call.call_id,
                        "output": json.dumps(
                            self.function_call_result
                        ),
                    }
                ],
            )

            self.runtime_calculator.end()

            print(
                f"API call duration: "
                f"{self.runtime_calculator.calculate_runtime():.2f} seconds"
            )

            if final_response.usage:
                self.token_counter.add_number_of_input_tokens(
                    final_response.usage.input_tokens
                )
                self.token_counter.add_number_of_output_tokens(
                    final_response.usage.output_tokens
                )

            return final_response

        except Exception as e:
            print(type(e))
            raise Exception(
                "Chat response could not be generated."
            ) from e

    def add_dynamic_prompting(self, filename, print_conversation=True):
        user_input = input("You: ")
        self._append_message(Message(role="user", content=user_input))
        response = self._execute_chat_completion_query(self.message_history)
        if response:
            assistant_message = response.output_text
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
            tools=function_list,
            validate=validate,
        )

        assistant_message = chat_response.output_text

        self._append_message(
            Message("assistant", assistant_message)
        )

        self.chat_file_writer.save_response(
            assistant_message,
            filename
        )

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

    @staticmethod
    def _messages_to_input(messages: List[Message]) -> List[Dict]:
        return [
            {
                "role": message.role,
                "content": message.content,
            }
            for message in messages
            if message.role in {"system", "user", "assistant"}
        ]

    @staticmethod
    def _get_responses_tool_definition(function: BaseFunction) -> Dict:
        return function.get_responses_tool_definition(strict=False)
