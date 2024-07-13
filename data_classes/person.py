import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Union, Dict

from dotenv import load_dotenv
from marshmallow import Schema, fields, post_load
from utils import file_reader_util


@dataclass
class Relation:
    relation_type: str
    related_person: Union[int, 'Person']

    def to_dict(self) -> Dict:
        return {
            'relation_type': self.relation_type,
            'related_person': self.related_person
        }


class RelationSchema(Schema):
    relation_type = fields.Str(required=True)
    related_person = fields.Int(required=True)

    @post_load
    def make_relation(self, data, **kwargs):
        return Relation(**data)


@dataclass
class Person:
    id: int
    name: str
    cognomen: Optional[str] = ""
    profession: Optional[str] = ""
    family_relations: List[Relation] = field(default_factory=list)
    legal_relationship: List[Relation] = field(default_factory=list)
    place_of_origin: Optional[str] = ""
    title: Optional[str] = ""

    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'name': self.name,
            'cognomen': self.cognomen,
            'profession': self.profession,
            'family_relations': [rel.to_dict() for rel in self.family_relations],
            'legal_relationship': [rel.to_dict() for rel in self.legal_relationship],
            'place_of_origin': self.place_of_origin,
            'title': self.title
        }


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
        return Person(**json_data)


def create_person_object(file_path: str) -> List[Person]:
    data = file_reader_util.read_json(file_path)

    person_schema = PersonSchema()
    persons = [person_schema.load(person) for person in data['person_list']]

    person_dict: Dict[int, Person] = {person.id: person for person in persons}

    for person in persons:
        for relation in person.family_relations + person.legal_relationship:
            if isinstance(relation.related_person, int):
                related_person = person_dict.get(relation.related_person)
                if related_person:
                    relation.related_person = related_person.name
                else:
                    relation.related_person = None

    return persons


if __name__ == "__main__":
    load_dotenv()
    base_dir = Path(os.getenv('PROJECT_BASE_DIR'))
    path_json = os.path.join(base_dir, "test_data", "test_json_diff", "sample_json_original.json")
    persons_in_json = create_person_object(path_json)

    for person_in_json in persons_in_json:
        print(person_in_json)
        for rel in person_in_json.family_relations + person_in_json.legal_relationship:
            print(f'  - {rel.relation_type} -> {rel.related_person.name}')
