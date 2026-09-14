import json
from typing import List, Literal
import openai
from jsons import ValidationError
from openai.types.responses import Response
from tenacity import retry, stop_after_attempt, wait_random_exponential
from evaluation.json_validation import JsonValidator
from function_calling_setup.chat_completion_blueprint import DialogueCompletion
from function_calling_setup.function_calling import Message
from function_definition.extract_json_from_plain_text import ExtractJsonFromPlainText


ResponseMode = Literal["text", "prompted_json", "json_schema"]


class DialogueCompletionNoFunctionCallingNew(DialogueCompletion):

    def __init__(self, model: str, experiment_dir):
        super().__init__(model=model, experiment_dir=experiment_dir)
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

        response = self._execute_response_query(
            messages=self.message_history,
            response_mode=response_mode,
            validate=validate,
        )

        assistant_message = response.output_text
        self._append_message(Message("assistant", assistant_message))
        self.chat_file_writer.save_response(assistant_message, filename)

        if print_conversation:
            self._print_conversation()

        return assistant_message

    @retry(
        stop=stop_after_attempt(6),
        wait=wait_random_exponential(multiplier=1, max=10),
        reraise=True,
    )
    def _execute_response_query(
            self,
            messages: List[Message],
            response_mode: ResponseMode,
            validate=True,
    ) -> Response:

        response = self._request_direct_response(messages=messages, response_mode=response_mode)

        if response_mode in {"prompted_json", "json_schema"}:
            self.json_result = self._parse_json_result(
                content=response.output_text,
                validate=validate,
            )

        return response

    def _request_direct_response(
            self,
            messages: List[Message],
            response_mode: ResponseMode,
    ) -> Response:

        input_messages = self._messages_to_input(messages)
        print(f"Input response content: {input_messages}")

        request = {
            "model": self.model,
            "input": input_messages,
            **self._get_reasoning_kwargs(),
        }

        if response_mode == "json_schema":
            schema = ExtractJsonFromPlainText().get_definition().function.parameters.to_schema()
            request["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": "person_list_schema",
                    "schema": schema,
                    "strict": True,
                }
            }

        elif response_mode not in {"text", "prompted_json"}:
            raise ValueError(f"Unknown response mode: {response_mode}")

        try:
            self.runtime_calculator.start()
            response = self.client.responses.create(**request)
            self.runtime_calculator.end()
            print(f"API call duration: {self.runtime_calculator.calculate_runtime():.2f} seconds")
            print(f"Output response content: {response.output_text}")

            if response.usage:
                self.token_counter.add_number_of_input_tokens(response.usage.input_tokens)
                self.token_counter.add_number_of_output_tokens(response.usage.output_tokens)

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

    @staticmethod
    def _remove_json_delimiters(content: str) -> str:
        return content.replace("```json", "").replace("```", "").strip()

    def _parse_json_result(self, content: str, validate=True):
        cleaned_content = self._remove_json_delimiters(content)
        print("This content will be parsed:", cleaned_content)

        json_output_as_dict = json.loads(cleaned_content)
        print("This is the JSON output as dict:", json_output_as_dict)

        function_object = ExtractJsonFromPlainText()
        json_result = function_object.run( **json_output_as_dict)
        print("This is the JSON result:", json_result)

        if validate:
            json_validator = JsonValidator()
            validation = json_validator.validate_json(json_result)

            if not validation[0]:
                raise ValidationError("JSON validation failed: " + validation[1])

        return json_result