import json
import os
from typing import cast
import openai
from openai import OpenAI
from openai.types.responses import EasyInputMessageParam, FunctionToolParam
from function_calling_setup.function_calling import Message
from function_calling_setup.providers.base_provider import LLMProvider
from function_calling_setup.providers.provider_response import ProviderResponse, ToolCall
from function_definition.base_function import BaseFunction
from function_calling_setup.models.model_config import ModelConfig


class OpenAIProvider(LLMProvider):

    def __init__(self, config: ModelConfig, strict: bool = False):
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.config = config
        self.model = config.model
        self.strict = strict

    def _get_reasoning_kwargs(self) -> dict:
        if self.config.reasoning_effort is None:
            return {}

        return {"reasoning": {"effort": self.config.reasoning_effort}}

    @staticmethod
    def _messages_to_input(messages: list[Message]) -> list[EasyInputMessageParam]:
        return [cast(EasyInputMessageParam, {"role": message.role, "content": message.content})
            for message in messages if message.role in {"system", "user", "assistant"}
        ]

    def _get_tool_definition(self, function: BaseFunction) -> FunctionToolParam:
        return function.get_responses_tool_definition(strict=self.strict)

    @staticmethod
    def _to_provider_response(response) -> ProviderResponse:
        tool_calls = []

        for item in response.output:
            if item.type == "function_call":
                tool_calls.append(
                    ToolCall(
                        name=item.name,
                        arguments=json.loads(item.arguments),
                        call_id=item.call_id,
                        raw=item,
                    )
                )

        input_tokens = 0
        output_tokens = 0

        if response.usage:
            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens

        return ProviderResponse(
            text=response.output_text,
            tool_calls=tool_calls,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            raw=response
        )

    def request(
            self,
            messages: list[Message],
            tools: list[BaseFunction] | None = None,
            schema: dict | None = None
    ) -> ProviderResponse:

        if tools and schema is not None:
            raise ValueError("Tools and direct JSON schema output cannot be requested together.")

        input_messages = self._messages_to_input(messages)

        request = {
            "model": self.model,
            "input": input_messages,
            **self._get_reasoning_kwargs(),
        }

        if tools:
            request["tools"] = [ self._get_tool_definition(function) for function in tools]
            request["tool_choice"] = "required"
            request["parallel_tool_calls"] = False

        if schema is not None:
            request["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": "person_list_schema",
                    "schema": schema,
                    "strict": True
                }
            }

        try:
            response = self.client.responses.create(**request)

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

        return self._to_provider_response(response)

    def submit_tool_result(
            self,
            messages: list[Message],
            previous_response: ProviderResponse,
            tool_call: ToolCall,
            tool_result: dict,
            tools: list[BaseFunction],
    ) -> ProviderResponse:

        if tool_call.call_id is None:
            raise ValueError("OpenAI tool call is missing a call_id.")

        raw_response = previous_response.raw

        response = self.client.responses.create(
            model=self.model,
            previous_response_id=raw_response.id,
            input=[
                {
                    "type": "function_call_output",
                    "call_id": tool_call.call_id,
                    "output": json.dumps(tool_result),
                }
            ],
            **self._get_reasoning_kwargs(),
        )

        return self._to_provider_response(response)