from typing import List, Tuple
from data_classes.person import Person


class PersonMatching:

    def __init__(
            self,
            matches: List[Tuple[Person, Person]] = None,
            not_found: List[Person] = None,
            unmatched_predictions: List[Person] = None,
    ):
        self.matches = matches if matches is not None else []
        self.not_found = not_found if not_found is not None else []
        self.unmatched_predictions = unmatched_predictions if unmatched_predictions is not None else []


