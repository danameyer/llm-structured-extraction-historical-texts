import json
from typing import Literal

from jsons import ValidationError
from tenacity import retry, stop_after_attempt, wait_random_exponential

from evaluation.json_validation import JsonValidator
from function_calling_setup.dialogue_completion import DialogueCompletion
from function_calling_setup.function_calling import Message
from function_calling_setup.providers.base_provider import LLMProvider
from function_calling_setup.providers.provider_response import ProviderResponse
from function_calling_setup.retry_tracking import MAX_ATTEMPTS
from function_definition.extract_json_from_plain_text import ExtractJsonFromPlainText


ResponseMode = Literal["text", "prompted_json", "json_schema"]


class DialogueCompletionNoFunctionCalling(DialogueCompletion):

    def __init__(
            self,
            provider: LLMProvider,
            experiment_dir,
    ):
        super().__init__(
            provider=provider,
            experiment_dir=experiment_dir,
        )
        self.json_result = None

    def prompt_assistant_response(
            self,
            prompt,
            filename,
            print_conversation=True,
            validate=True,
            response_mode: ResponseMode = "text",
    ):
        self._append_message(Message("user", prompt))
        self.chat_file_writer.save_prompt(prompt, filename)

        if response_mode in {"prompted_json", "json_schema"}:
            self.json_result = None
            self.retry_tracker.reset()

        response = self._execute_response_query(
            messages=self.message_history,
            response_mode=response_mode,
            validate=validate,
        )

        assistant_message = response.text

        self._append_message(
            Message("assistant", assistant_message)
        )
        self.chat_file_writer.save_response(
            assistant_message,
            filename,
        )

        if print_conversation:
            self._print_conversation()

        return assistant_message

    @retry(
        stop=stop_after_attempt(MAX_ATTEMPTS),
        wait=wait_random_exponential(multiplier=1, max=10),
        reraise=True,
    )
    def _execute_response_query(
            self,
            messages: list[Message],
            response_mode: ResponseMode,
            validate: bool = True,
    ) -> ProviderResponse:

        is_extraction_request = response_mode in {
            "prompted_json",
            "json_schema",
        }

        if is_extraction_request:
            self.retry_tracker.record_attempt()

        schema = None

        if response_mode == "json_schema":
            schema = (
                ExtractJsonFromPlainText()
                .get_definition()
                .function
                .parameters
                .to_schema()
            )

        elif response_mode not in {
            "text",
            "prompted_json",
        }:
            raise ValueError(
                f"Unknown response mode: {response_mode}"
            )

        response = self._request_response(
            messages=messages,
            schema=schema,
        )

        if is_extraction_request:
            self.json_result = self._parse_json_result(
                content=response.text,
                validate=validate,
            )
            self.retry_tracker.mark_success()

        return response

    @staticmethod
    def _remove_json_delimiters(content: str) -> str:
        return (
            content
            .replace("```json", "")
            .replace("```", "")
            .strip()
        )

    def _parse_json_result(
            self,
            content: str,
            validate: bool = True,
    ):
        cleaned_content = self._remove_json_delimiters(
            content
        )

        print(
            "This content will be parsed:",
            cleaned_content,
        )

        json_output_as_dict = json.loads(
            cleaned_content
        )

        print(
            "This is the JSON output as dict:",
            json_output_as_dict,
        )

        function_object = ExtractJsonFromPlainText()

        json_result = function_object.run(
            **json_output_as_dict
        )

        print(
            "This is the JSON result:",
            json_result,
        )

        if validate:
            json_validator = JsonValidator()
            validation = json_validator.validate_json(
                json_result
            )

            if not validation[0]:
                raise ValidationError(
                    "JSON validation failed: "
                    + validation[1]
                )

        return json_result