import json
import os
import httpx
from function_calling_setup.function_calling import Message
from function_calling_setup.models.model_config import ModelConfig
from function_calling_setup.providers.base_provider import LLMProvider
from function_calling_setup.providers.provider_response import ProviderResponse, ToolCall
from function_definition.base_function import BaseFunction


class OllamaProvider(LLMProvider):

    def __init__(self, config: ModelConfig):
        base_url = os.environ.get("OLLAMA_URL")
        token = os.environ.get("OLLAMA_TOKEN")

        if not base_url:
            raise ValueError("OLLAMA_URL is not set.")

        if not token:
            raise ValueError("OLLAMA_TOKEN is not set.")

        self.base_url = base_url.rstrip("/")
        self.config = config
        self.model = config.model
        timeout_seconds = float(os.environ.get("OLLAMA_TIMEOUT_SECONDS", "600"))
        self.timeout = httpx.Timeout(timeout_seconds, connect=30.0)
        self.headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    def _get_api_chat_thinking_kwargs(self) -> dict:
        if self.config.think is None:
            return {}

        return {"think": self.config.think}

    def _get_chat_completions_reasoning_kwargs(self) -> dict:
        if self.config.think is None:
            return {}

        if self.config.think is False:
            return {"reasoning_effort": "none"}

        if self.config.think is True:
            return {"reasoning_effort": "medium"}

        return {"reasoning_effort": self.config.think,}

    @staticmethod
    def _convert_messages_to_input(messages: list[Message]) -> list[dict]:
        return [{"role": message.role, "content": message.content}
            for message in messages
            if message.role in {"system", "user", "assistant"}
        ]

    @staticmethod
    def _get_tool_definition(function: BaseFunction) -> dict:
        definition = function.get_definition()

        return {
            "type": "function",
            "function": {
                "name": definition.function.name,
                "description": definition.function.description,
                "parameters": definition.function.parameters.to_schema(),
            }
        }

    def _send_post_request(self, endpoint: str, payload: dict) -> dict:
        try:
            response = httpx.post(
                f"{self.base_url}{endpoint}",
                json=payload,
                headers=self.headers,
                timeout=self.timeout,
            )

            response.raise_for_status()

        except httpx.HTTPStatusError as error:
            raise RuntimeError(f"Ollama returned HTTP {error.response.status_code}: {error.response.text}") from error

        except httpx.RequestError as error:
            raise RuntimeError(f"Could not reach the Ollama server at {self.base_url}.") from error

        return response.json()

    @staticmethod
    def _parse_tool_arguments(arguments) -> dict:
        if arguments is None:
            return {}

        if isinstance(arguments, dict):
            return arguments

        if isinstance(arguments, str):
            parsed = json.loads(arguments)

            if isinstance(parsed, dict):
                return parsed

        raise ValueError("Ollama returned invalid tool calling arguments.")

    @classmethod
    def _parse_api_chat_response(cls, response: dict) -> ProviderResponse:
        message = response.get("message") or {}
        tool_calls = []

        for item in message.get("tool_calls") or []:
            function = item.get("function") or {}
            name = function.get("name")

            if not name:
                raise ValueError("Ollama returned a tool call without a name.")

            tool_calls.append(
                ToolCall(
                    name=name,
                    arguments=cls._parse_tool_arguments(function.get("arguments")),
                    call_id=item.get("id"),
                    raw=item,
                )
            )

        return ProviderResponse(
            text=message.get("content") or "",
            reasoning=message.get("thinking") or message.get("reasoning") or "",
            tool_calls=tool_calls,
            input_tokens=response.get("prompt_eval_count", 0) or 0,
            output_tokens=response.get("eval_count", 0) or 0,
            raw=response
        )

    @classmethod
    def _parse_chat_completions_response(cls, response: dict) -> ProviderResponse:
        choices = response.get("choices") or []

        if not choices:
            raise ValueError("Ollama returned no completion choices.")

        message = choices[0].get("message") or {}
        tool_calls = []

        for item in message.get("tool_calls") or []:
            function = item.get("function") or {}
            name = function.get("name")

            if not name:
                raise ValueError("Ollama returned a tool call without a name.")

            tool_calls.append(
                ToolCall(
                    name=name,
                    arguments=cls._parse_tool_arguments(function.get("arguments")),
                    call_id=item.get("id"),
                    raw=item
                )
            )

        usage = response.get("usage") or {}

        return ProviderResponse(
            text=message.get("content") or "",
            reasoning=message.get("reasoning") or message.get("reasoning_content") or message.get("thinking") or "",
            tool_calls=tool_calls,
            input_tokens=usage.get("prompt_tokens", 0) or 0,
            output_tokens=usage.get("completion_tokens", 0) or 0,
            raw=response
        )

    def request(
            self,
            messages: list[Message],
            tools: list[BaseFunction] | None = None,
            schema: dict | None = None,
    ) -> ProviderResponse:

        if tools and schema is not None:
            raise ValueError("Tools and direct JSON schema output cannot be requested together.")

        if tools:
            reasoning_kwargs = self._get_chat_completions_reasoning_kwargs()
            print("Ollama endpoint: /v1/chat/completions")
            print(f"Ollama reasoning kwargs: {reasoning_kwargs}")

            payload = {
                "model": self.model,
                "messages": self._convert_messages_to_input(messages),
                "tools": [self._get_tool_definition(function) for function in tools],
                "tool_choice": "required",
                "stream": False,
                **reasoning_kwargs
            }

            response = self._send_post_request("/v1/chat/completions", payload)
            choices = response.get("choices") or []

            if choices:
                choice = choices[0]
                message = choice.get("message") or {}
                print("Ollama finish reason:", choice.get("finish_reason"))
                print("Ollama response message keys:", sorted(message.keys()))
                print("Ollama raw tool calls:", message.get("tool_calls"))
                print("Ollama content preview:", repr((message.get("content") or "")[:500]))

            provider_response = self._parse_chat_completions_response(response)
            print(
                f"Ollama reasoning output: present={bool(provider_response.reasoning)}, "
                f"chars={len(provider_response.reasoning)}"
            )

            return provider_response

        thinking_kwargs = self._get_api_chat_thinking_kwargs()
        print("Ollama endpoint: /api/chat")
        print(f"Ollama thinking kwargs: {thinking_kwargs}")

        payload = {
            "model": self.model,
            "messages": self._convert_messages_to_input(messages),
            "stream": False,
            **thinking_kwargs
        }

        if schema is not None:
            payload["format"] = schema

        response = self._send_post_request("/api/chat", payload)
        provider_response = self._parse_api_chat_response(response)
        print(f"Ollama API chat reasoning output: {provider_response.reasoning!r}")

        return provider_response

    def submit_tool_result(
            self,
            messages: list[Message],
            previous_response: ProviderResponse,
            tool_call: ToolCall,
            tool_result: dict,
            tools: list[BaseFunction],
    ) -> ProviderResponse:

        if tool_call.call_id is None:
            raise ValueError("Ollama tool call is missing a call_id.")

        assistant_message = {
            "role": "assistant",
            "content": previous_response.text,
            "tool_calls": [
                {
                    "id": tool_call.call_id,
                    "type": "function",
                    "function": {"name": tool_call.name, "arguments": json.dumps(tool_call.arguments)}
                }
            ]
        }

        conversation = self._convert_messages_to_input(messages)
        conversation.append(assistant_message)
        conversation.append(
            {
                "role": "tool",
                "tool_call_id": tool_call.call_id,
                "content": json.dumps(tool_result)
            }
        )

        reasoning_kwargs = self._get_chat_completions_reasoning_kwargs()
        print("Ollama tool-result endpoint: /v1/chat/completions")
        print(f"Ollama tool-result reasoning kwargs: {reasoning_kwargs}")

        payload = {
            "model": self.model,
            "messages": conversation,
            "stream": False,
            **reasoning_kwargs
        }

        response = self._send_post_request("/v1/chat/completions", payload)
        provider_response = self._parse_chat_completions_response(response)
        print(f"Ollama chat completions reasoning output after tool call: {provider_response.reasoning!r}")

        return provider_response
