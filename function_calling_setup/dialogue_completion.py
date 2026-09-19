import json
from typing import Dict, List, Optional
from jsons import ValidationError
from tenacity import retry, stop_after_attempt, wait_random_exponential
from evaluation.json_validation import JsonValidator
from function_calling_setup.chat_file_writer import ChatFileWriter
from function_calling_setup.function_calling import Message
from function_calling_setup.providers.provider_response import ProviderResponse, ToolCall
from function_calling_setup.retry_tracking import MAX_ATTEMPTS, RetryTracker
from function_calling_setup.runtime_calculation import RuntimeCalculation
from function_calling_setup.models.token_counting import TokenCounter
from function_definition.base_function import BaseFunction
from function_calling_setup.providers.base_provider import LLMProvider

class DialogueCompletion:

    def __init__(self, provider: LLMProvider, experiment_dir):
        self.provider = provider
        self.message_history: list[Message] = []
        self.chat_file_writer = ChatFileWriter(experiment_dir)
        self.function_call_result = None
        self.token_counter = TokenCounter(provider.model)
        self.runtime_calculator = RuntimeCalculation()
        self.retry_tracker = RetryTracker()

    def _request_response(
            self,
            messages: list[Message],
            functions: list[BaseFunction] | None = None,
            schema: dict | None = None,
    ) -> ProviderResponse:

        input_messages = [{"role": message.role, "content": message.content} for message in messages]
        print(f"Input response content: {input_messages}")

        self.runtime_calculator.start()

        try:
            response = self.provider.request(
                messages=messages,
                tools=functions,
                schema=schema,
            )
        finally:
            self.runtime_calculator.end()

        print(f"API call duration: {self.runtime_calculator.calculate_runtime():.2f} seconds")
        print(f"Output response content: {response.text}")

        self.token_counter.add_number_of_input_tokens(response.input_tokens)
        self.token_counter.add_number_of_output_tokens(response.output_tokens)

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

        response = self._request_response(messages=messages, functions=tools)

        if response.tool_calls:
            print("Function will be called.")

            if not tools:
                raise ValueError("Model returned a tool call although no tools were provided.")

            result = self._perform_function_call(
                response=response,
                function_call=response.tool_calls[0],
                tools=tools,
                validate=validate
            )

            self.retry_tracker.mark_success()
            return result

        if tools:
            raise ValueError("Expected a function call, but the model did not return one.")

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
        print("These are the function call arguments:", function_parameters)

        function_object = next(
            (
                function for function in tools
                if function.get_definition().function.name == function_name
            ),
            None
        )

        if function_object is None:
            raise ValueError(f"Unknown function requested: {function_name}")

        self.function_call_result = function_object.run(**function_parameters)

        print("This is the function call result", self.function_call_result)

        if validate:
            json_validator = JsonValidator()
            validation = json_validator.validate_json(self.function_call_result)

            if not validation[0]:
                raise ValidationError("JSON validation failed: " + validation[1])

        return response

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

        if function_list:
            assistant_message = json.dumps(self.function_call_result, ensure_ascii=False, indent=2)
        else:
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
