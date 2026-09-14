import json
import os
from pathlib import Path
import jsonschema
from dotenv import load_dotenv
from jsonschema.validators import Draft202012Validator

from utils import file_reader_util


class JsonValidator:
    def __init__(self):
        load_dotenv()

        self.json_schema = {
            "type": "object",
            "properties": {
                "person_list": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "number"},
                            "name": {"type": "string"},
                            "cognomen": {"type": "string"},
                            "profession": {"type": "string"},
                            "family_relations": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "relation_type": {"type": "string"},
                                        "related_person": {"type": ["number", "null"]}
                                    },
                                    "required": ["relation_type", "related_person"]
                                }
                            },
                            "legal_relationship": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "relation_type": {"type": "string"},
                                        "related_person": {"type": ["number", "null"]}
                                    },
                                    "required": ["relation_type", "related_person"]
                                }
                            },
                            "place_of_origin": {"type": "string"},
                            "title": {"type": "string"}
                        },
                        "required": ["id", "name", "cognomen", "profession", "family_relations", "legal_relationship",
                                     "place_of_origin", "title"]
                    }
                }
            },
            "required": ["person_list"]
        }

    def validate_json(self, json_data):
        try:
            if isinstance(json_data, str):
                data = json.loads(json_data)
            else:
                data = json_data

            print("Data being validated:", data)

            json_validator = Draft202012Validator(self.json_schema)
            errors = sorted(json_validator.iter_errors(data), key=lambda exception: exception.path)

            if errors:
                for error in errors:
                    message = f"Validation error at json-path {list(error.path)} with message: {error.message}"
                    print(message)
                    return False, message
            else:
                print("JSON is valid and matches the schema.")
                return True, "Json is valid."
        except json.JSONDecodeError as e:
            print(f"Invalid JSON data: {e}")
        except jsonschema.exceptions.ValidationError as e:
            print(f"JSON does not match schema: {e}")
