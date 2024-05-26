import json
from typing import Any, Dict

import jsons
from dotenv import load_dotenv

from function_calling_components.function_calling import FunctionBuilder


class BaseFunction:

    def __init__(self):
        load_dotenv()
        self.use_short_definition = False

    def flag_use_short_definition_true(self):
        self.use_short_definition = True

    def get_definition(self) -> FunctionBuilder:
        pass

    def get_definition_dict(self) -> Dict:
        if self.use_short_definition:
            return {"name": self.get_definition().name}
        else:
            return json.loads(jsons.dumps(self.get_definition()))

    def run(self, name) -> Any:
        pass
