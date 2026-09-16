import json
import os
from typing import TypedDict, Literal, List, Optional, cast, Dict
import openai
from dotenv import load_dotenv
from jsons import ValidationError
from openai import OpenAI
from evaluation.json_validation import JsonValidator
from function_calling_setup.chat_file_writer import ChatFileWriter
from function_calling_setup.function_calling import Message
from function_calling_setup.providers.openai_provider import _get_reasoning_kwargs, OpenAIProvider
from function_calling_setup.providers.provider_response import ProviderResponse, ToolCall
from function_calling_setup.runtime_calculation import RuntimeCalculation
from function_calling_setup.token_counting import TokenCounter
from function_definition.base_function import BaseFunction
from tenacity import retry, stop_after_attempt, wait_random_exponential
from openai.types.responses import EasyInputMessageParam, FunctionToolParam, Response
from openai.types.shared_params.reasoning import Reasoning
from function_calling_setup.retry_tracking import MAX_ATTEMPTS, RetryTracker

load_dotenv()

ToolChoiceMode = Literal["none", "auto", "required"]

class DialogueCompletion:

    def __init__(self, model: str, experiment_dir, strict: bool = False):
        load_dotenv()
        self.provider = OpenAIProvider(model=model, strict=strict)
        self.message_history: list[Message] = []
        self.chat_file_writer = ChatFileWriter(experiment_dir)
        self.function_call_result = None
        self.token_counter = TokenCounter(model)
        self.runtime_calculator = RuntimeCalculation()
        self.retry_tracker = RetryTracker()

    def _request_response(
            self,
            messages: list[Message],
            functions: list[BaseFunction] | None = None,
    ) -> ProviderResponse:

        print(f"Input response content: {messages}")

        self.runtime_calculator.start()

        response = self.provider.request(
            messages=messages,
            tools=functions,
        )

        self.runtime_calculator.end()

        print(
            f"API call duration: "
            f"{self.runtime_calculator.calculate_runtime():.2f} seconds"
        )

        print(
            f"Output response content: {response.text}"
        )

        self.token_counter.add_number_of_input_tokens(
            response.input_tokens
        )
        self.token_counter.add_number_of_output_tokens(
            response.output_tokens
        )

        return response

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
        stop=stop_after_attempt(MAX_ATTEMPTS),
        wait=wait_random_exponential(multiplier=1, max=10),
        reraise=True
    )
    def _execute_chat_completion_query(
            self,
            messages: List[Message],
            tools: Optional[List[BaseFunction]] = None,
            validate=True,
    ) -> ProviderResponse:

        if tools:
            self.retry_tracker.record_attempt()

        response = self._request_response(
            messages=messages,
            functions=tools,
        )

        if response.tool_calls:
            print("Function will be called.")

            result = self._perform_function_call(
                response=response,
                function_call=response.tool_calls[0],
                tools=tools,
                validate=validate,
            )

            self.retry_tracker.mark_success()

            return result

        if tools:
            raise ValueError(
                "Expected a function call, "
                "but the model did not return one."
            )

        print("No function called.")
        return response

    def _perform_function_call(
            self,
            response: ProviderResponse,
            function_call: ToolCall,
            tools: list[BaseFunction],
            validate: bool = True,
    ) -> ProviderResponse:

        function_name = function_call.name
        function_parameters = function_call.arguments

        print(
            "These are the function call arguments:",
            function_parameters,
        )

        function_object = next(
            (
                function
                for function in tools
                if function.get_definition().function.name == function_name
            ),
            None,
        )

        if function_object is None:
            raise ValueError(
                f"Unknown function requested: {function_name}"
            )

        self.function_call_result = function_object.run(
            **function_parameters
        )

        print(
            "This is the function call result",
            self.function_call_result,
        )

        if validate:
            json_validator = JsonValidator()
            validation = json_validator.validate_json(
                self.function_call_result
            )

            if not validation[0]:
                raise ValidationError(
                    "JSON validation failed: " + validation[1]
                )

        self.runtime_calculator.start()

        final_response = self.provider.submit_tool_result(
            previous_response=response,
            tool_call=function_call,
            tool_result=self.function_call_result,
        )

        self.runtime_calculator.end()

        print(
            f"API call duration: "
            f"{self.runtime_calculator.calculate_runtime():.2f} seconds"
        )

        self.token_counter.add_number_of_input_tokens(
            final_response.input_tokens
        )
        self.token_counter.add_number_of_output_tokens(
            final_response.output_tokens
        )

        return final_response

    def add_dynamic_prompting(self, filename, print_conversation=True):
        user_input = input("You: ")
        self._append_message(Message(role="user", content=user_input))
        response = self._execute_chat_completion_query(self.message_history)
        if response:
            assistant_message = response.text
            self._append_message(Message(role="assistant", content=assistant_message))
            if print_conversation:
                self._print_conversation()

            self.chat_file_writer.save_response(f"User: {user_input}", filename)
            self.chat_file_writer.save_response(f"Assistant: {assistant_message}", filename)

    def prompt_assistant_response(
            self,
            prompt,
            filename,
            function_list=None,
            print_conversation=True,
            validate=True,
    ):
        self._append_message(Message("user", prompt))
        self.chat_file_writer.save_prompt(prompt, filename)

        if function_list:
            self.function_call_result = None
            self.retry_tracker.reset()

        chat_response = self._execute_chat_completion_query(
            messages=self.message_history,
            tools=function_list,
            validate=validate
        )

        assistant_message = chat_response.text

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

    @staticmethod
    def flag_function_calls_for_short_description(functions: List[BaseFunction] | None) -> None:
        if functions:
            for function in functions:
                function.flag_use_short_definition_true()

