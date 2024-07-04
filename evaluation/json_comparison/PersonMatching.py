import json
from typing import List, Tuple, Dict

from langchain.evaluation import JsonEditDistanceEvaluator


class PersonMatching:

    def __init__(self, matches: List[Tuple[Dict, Dict]] = None, not_found:  List[Dict] = None):

        if matches is None:
            self.matches = list()
        else:
            self.matches: List[Tuple[Dict, Dict]] = matches

        if not_found is None:
            self.not_found = list()
        else:
            self.not_found: List[Dict] = not_found


