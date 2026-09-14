import os
from dataclasses import field, dataclass
from pathlib import Path
from typing import List
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
