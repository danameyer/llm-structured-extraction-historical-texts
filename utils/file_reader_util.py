import json


def read_json(json_file):
    with open(json_file, 'r') as file:
        json_content = file.read()
    json_data = json.loads(json_content)
    return json_data


def read_file(filepath):
    with open(filepath, 'r') as file:
        content = file.read().splitlines()
    return content
