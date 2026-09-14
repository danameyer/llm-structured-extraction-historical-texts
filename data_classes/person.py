import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict
import marshmallow
from dotenv import load_dotenv
from marshmallow import Schema, fields, post_load
from utils import file_reader_util


@dataclass
class Relation:
    relation_type: str
    related_person: int | None

    def to_dict(self) -> Dict:
        return {'relation_type': self.relation_type, 'related_person': self.related_person}


class RelationSchema(Schema):
    relation_type = fields.Raw(required=True)
    related_person = fields.Raw(required=True, allow_none=True)

    @post_load
    def make_relation(self, data, **kwargs):
        return Relation(**data)


@dataclass
class Person:
    def __init__(self, person_id=None, name="", cognomen="", profession="", family_relations=None,
                 legal_relationship=None, place_of_origin="", title=""):
        self.id = person_id
        self.name = name
        self.cognomen = cognomen
        self.profession = profession
        self.family_relations = family_relations or []
        self.legal_relationship = legal_relationship or []
        self.place_of_origin = place_of_origin
        self.title = title

    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'name': self.name,
            'cognomen': self.cognomen,
            'profession': self.profession,
            'family_relations': [relation.to_dict() for relation in self.family_relations],
            'legal_relationship': [relation.to_dict() for relation in self.legal_relationship],
            'place_of_origin': self.place_of_origin,
            'title': self.title
        }

    def __eq__(self, other):
        if isinstance(other, Person):
            return (
                    self.id == other.id and
                    self.name == other.name and
                    self.cognomen == other.cognomen and
                    self.profession == other.profession and
                    self.family_relations == other.family_relations and
                    self.legal_relationship == other.legal_relationship and
                    self.place_of_origin == other.place_of_origin and
                    self.title == other.title
            )
        return False


class PersonSchema(Schema):
    id = fields.Int(required=True)
    name = fields.Str(required=True)
    cognomen = fields.Str(load_default="")
    profession = fields.Str(load_default="")
    family_relations = fields.List(fields.Nested(RelationSchema()), load_default=[])
    legal_relationship = fields.List(fields.Nested(RelationSchema()), load_default=[])
    place_of_origin = fields.Str(load_default="")
    title = fields.Str(load_default="")

    @post_load
    def make_person(self, json_data, **kwargs):
        person_data = json_data.copy()
        person_id = person_data.pop("id")

        return Person(person_id=person_id, **person_data)


def create_person_object(file_path: str) -> List[Person]:
    data = file_reader_util.read_json(file_path)

    if data and data.get("person_list") is not None:
        print(data["person_list"])
    else:
        print("person_list is None or not present.")

    person_schema = PersonSchema()

    if data is not None and "person_list" in data:
        try:
            persons = [person_schema.load(person) for person in data["person_list"] if person is not None]
        except marshmallow.exceptions.ValidationError as ex:
            logging.error(f"Could not load PersonSchema for file {file_path} because of exception: {ex.messages}")
            persons = []
    else:
        persons = []

    return persons


if __name__ == "__main__":
    load_dotenv()
    base_dir = Path(os.getenv('PROJECT_BASE_DIR'))
    path_json = os.path.join(base_dir, "test_data", "test_json_diff", "sample_json_original.json")
    persons_in_json = create_person_object(path_json)

    for person_in_json in persons_in_json:
        print(person_in_json)
        for rel in person_in_json.family_relations + person_in_json.legal_relationship:
            print(f"  - {rel.relation_type} -> {rel.related_person}")
