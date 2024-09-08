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
            return {"name": self.get_definition().function.name}
        else:
            return json.loads(jsons.dumps(self.get_definition()))

    def get_function_dict_for_fine_tuning(self) -> Dict:
        return json.loads(jsons.dumps(self.get_definition().function))

    def del_none(self, d):
        """
        Delete keys with the value ``None`` in a dictionary, recursively.

        This alters the input so you may wish to ``copy`` the dict first.
        """
        # For Python 3, write `list(d.items())`; `d.items()` won’t work
        # For Python 2, write `d.items()`; `d.iteritems()` won’t work
        for key, value in list(d.items()):
            if value is None:
                del d[key]
            elif isinstance(value, dict):
                self.del_none(value)
        return d  # For convenience

    def run(self, name) -> Any:
        pass
