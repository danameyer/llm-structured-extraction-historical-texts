from jsonschema.validators import Draft202012Validator
from function_definition.extract_json_from_plain_text import ExtractJsonFromPlainText


class JsonValidator:

    def __init__(self):
        self.json_schema = ExtractJsonFromPlainText().get_internal_schema()

    def validate_json(self, json_data):
        validator = Draft202012Validator(self.json_schema)
        errors = sorted(validator.iter_errors(json_data), key=lambda exception: exception.path)

        if errors:
            error = errors[0]
            message = f"Validation error at json-path {list(error.path)} with message: {error.message}"
            print(message)
            return False, message

        print("JSON is valid and matches the schema."
)
        return True, "JSON is valid."