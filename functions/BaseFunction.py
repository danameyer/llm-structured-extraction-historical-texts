import json
from typing import Any, Dict

import jsons

from function_calling_components.function_calling import FunctionBuilder


class BaseFunction:

    def get_definition(self) -> FunctionBuilder:
        pass

    def get_definition_dict(self) -> Dict:
        return json.loads(jsons.dumps(self.get_definition()))

    def run(self, name) -> Any:
        pass
