from abc import ABC, abstractmethod
from function_calling_setup.function_calling import Message
from function_calling_setup.providers.provider_response import ProviderResponse, ToolCall
from function_definition.base_function import BaseFunction


class LLMProvider(ABC):
    model: str

    @abstractmethod
    def request(
            self,
            messages: list[Message],
            tools: list[BaseFunction] | None = None,
            schema: dict | None = None,
    ) -> ProviderResponse:
        pass

    @abstractmethod
    def submit_tool_result(
            self,
            messages: list[Message],
            previous_response: ProviderResponse,
            tool_call: ToolCall,
            tool_result: dict,
            tools: list[BaseFunction],
    ) -> ProviderResponse:
        pass