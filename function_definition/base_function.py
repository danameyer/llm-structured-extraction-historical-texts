from typing import Any, Dict
from function_calling_setup.function_calling import FunctionBuilder


class BaseFunction:

    def __init__(self):
        pass

    def get_definition(self) -> FunctionBuilder:
        pass

    def get_responses_tool_definition(self, strict: bool = False) -> Dict:
        definition = self.get_definition()

        return {
            "type": "function",
            "name": definition.function.name,
            "description": definition.function.description,
            "parameters": definition.function.parameters.to_schema(),
            "strict": strict,
        }

    def del_none(self, d):
        """
        Delete keys with the value ``None`` in a dictionary, recursively.

        This alters the input so you may wish to ``copy`` the dict first.
        """
        for key, value in list(d.items()):
            if value is None:
                del d[key]
            elif isinstance(value, dict):
                self.del_none(value)
        return d

    def run(self, name) -> Any:
        pass
