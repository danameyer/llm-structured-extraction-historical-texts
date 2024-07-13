import json
from typing import List, Tuple, Dict

from langchain.evaluation import JsonEditDistanceEvaluator

from data_classes.person import Person


class PersonMatching:

    def __init__(self, matches: List[Tuple[Person, Person]] = None, not_found: List[Person] = None):

        if matches is None:
            self.matches = list()
        else:
            self.matches: List[Tuple[Person, Person]] = matches

        if not_found is None:
            self.not_found = list()
        else:
            self.not_found: List[Person] = not_found


