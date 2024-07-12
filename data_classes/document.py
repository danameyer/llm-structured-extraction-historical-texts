import os
from dataclasses import field, dataclass
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from marshmallow import post_load, fields, Schema

from data_classes.person import PersonSchema, Person, create_person_object


@dataclass
class Document:
    document_name: str
    persons: List[Person] = field(default_factory=list)


class DocumentSchema(Schema):
    document_name = fields.Str(required=True)
    persons = fields.List(fields.Nested(PersonSchema()), required=True)

    @post_load
    def make_document(self, data, **kwargs):
        return Document(**data)


def create_documents(folder_path: str) -> List[Document]:
    documents = []
    for json_file in os.listdir(folder_path):
        if json_file.endswith(".json"):
            file_path = os.path.join(folder_path, json_file)
            persons = create_person_object(file_path)
            document_name = Path(file_path).stem + '.json'
            document = Document(document_name=document_name, persons=persons)
            documents.append(document)
    return documents


if __name__ == "__main__":
    load_dotenv()
    base_dir = Path(os.getenv('PROJECT_BASE_DIR'))
    json_dir = os.path.join(base_dir, "test_data", "test_json_diff", "ground_truth")
    json_documents = create_documents(json_dir)

    for json_document in json_documents:
        print(f'Document: {json_document.document_name}')
        for person in json_document.persons:
            print(person)
            for rel in person.family_relations + person.legal_relationship:
                print(f'  - {rel.relation_type} -> {rel.related_person.name}')
