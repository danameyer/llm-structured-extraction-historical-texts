import heapq
import json
import os
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from langchain.evaluation import JsonEditDistanceEvaluator

from data_classes.document import Document, create_documents
from data_classes.person import Person
from evaluation.json_comparison.json_comparison import JsonComparison


class PersonComparison:
    def __init__(self):
        pass

    @staticmethod
    def find_best_persons_in_documents(documents: List[Document],
                                       person_to_find: Person, n: int = 5) -> List[Person]:
        """Find and return the n best matching Person objects."""
        evaluator = JsonEditDistanceEvaluator()
        json_comparison = JsonComparison()
        person_to_find_name = person_to_find.name
        best_matches = []

        for document in documents:
            for person_in_doc in document.persons:
                person_in_doc_name = person_in_doc.name
                if json_comparison.fuzzy_compare(person_to_find_name, person_in_doc_name):
                    person_to_find_as_dict = person_to_find.to_dict()
                    person_in_doc_as_dict = person_in_doc.to_dict()
                    person_to_find_as_string = json.dumps(person_to_find_as_dict)
                    person_in_doc_as_string = json.dumps(person_in_doc_as_dict)
                    result = evaluator.evaluate_strings(prediction=person_to_find_as_string,
                                                        reference=person_in_doc_as_string)
                    score = result['score']
                    heapq.heappush(best_matches, (score, person_in_doc))
                    if len(best_matches) > n:
                        heapq.heappop(best_matches)

        best_matches = [heapq.heappop(best_matches)[1] for _ in range(len(best_matches))]
        best_matches.reverse()
        return best_matches


if __name__ == '__main__':
    load_dotenv()
    base_dir = Path(os.getenv('PROJECT_BASE_DIR'))
    pred_folder = os.path.join(base_dir, "test_data", "test_json_diff", "predictions")
    gt_documents = create_documents(pred_folder)

    person_comparator = PersonComparison()

    sample_person = Person(
        id=1,
        name="Simon",
        cognomen="De Leukenor",
        profession="",
        family_relations=[],
        legal_relationship=[],
        place_of_origin="Leukenor",
        title=""
    )

    top_matches = person_comparator.find_best_persons_in_documents(gt_documents, sample_person, n=5)

    for match in top_matches:
        print(match)
