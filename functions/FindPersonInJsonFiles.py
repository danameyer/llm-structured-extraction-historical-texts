import os
from pathlib import Path

from dotenv import load_dotenv

from data_classes.document import create_documents
from data_classes.person import Person
from evaluation.json_comparison.compare_persons_across_documents import PersonComparison
from function_calling_components.function_calling import FunctionBuilder, Parameter, Property
from functions.BaseFunction import BaseFunction


class PersonFinder(BaseFunction):

    def get_definition(self) -> FunctionBuilder:
        return FunctionBuilder(
            name="find_person_in_json_files",
            description="Extract the information about the person from the user input",
            parameters=Parameter(
                parameter_type="object",
                properties={
                    "person_list": Property(
                        property_type="array",
                        description="List of people mentioned in the input plain text",
                        items=Property(
                            property_type="object",
                            description="Description of individual person mentioned in the input plain text",
                            properties={
                                "id": Property(property_type="integer", description="Unique identifier for the person"),
                                "name": Property(property_type="string", description="Name of the person"),
                                "cognomen": Property(property_type="string", description="Addition to first name", default_value=""),
                                "profession": Property(property_type="string", description="Profession of the person", default_value=""),
                                "family_relations": Property(
                                    property_type="array",
                                    description="List of family relations",
                                    items=Property(
                                        property_type="object",
                                        description="Family relation object",
                                        properties={
                                            "relation_type": Property(property_type="string", description="Type of family relation (e.g., pater, frater, filius, filia)", default_value=""),
                                            "related_person": Property(property_type="integer", description="ID of the related person", default_value=None)
                                        }
                                    ),
                                    default_value=[]
                                ),
                                "legal_relationship": Property(
                                    property_type="array",
                                    description="List of legal relations such as custos and heres",
                                    items=Property(
                                        property_type="object",
                                        description="Legal relationship object",
                                        properties={
                                            "relation_type": Property(property_type="string", description="Type of legal relation", default_value=""),
                                            "related_person": Property(property_type="integer", description="ID of the related person", default_value=None)
                                        }
                                    ),
                                    default_value=[]
                                ),
                                "place_of_origin": Property(property_type="string", description="Place of origin of the person", default_value=""),
                                "title": Property(property_type="string", description="Title of the person", default_value="")
                            }
                        )
                    )
                },
                required=["name"]
            )
        )

    def run(self, **kwargs):
        person_list = kwargs.get('person_list', [])
        if not person_list:
            return "No person information provided."

        person_to_find = person_list[0]

        load_dotenv()
        base_dir = Path(os.getenv('PROJECT_BASE_DIR'))
        pred_folder = os.path.join(base_dir, "test_data", "test_json_diff", "predictions")
        gt_documents = create_documents(pred_folder)

        person_comparator = PersonComparison()
        person = Person(
            name=person_to_find['name'],
            cognomen=person_to_find.get('cognomen', ''),
            profession=person_to_find.get('profession', ''),
            family_relations=person_to_find.get('family_relations', []),
            legal_relationship=person_to_find.get('legal_relationship', []),
            place_of_origin=person_to_find.get('place_of_origin', ''),
            title=person_to_find.get('title', '')
        )

        top_matches = person_comparator.find_best_persons_in_documents(gt_documents, person, n=5)

        top_matches_dicts = [{
            **match[1].to_dict(),
            'document_name': match[2]
        } for match in top_matches]

        return top_matches_dicts

