import json
import os
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv


class OpenAIFineTuningSchemaCheck:
    def __init__(self):
        pass

    def evaluate_open_ai_fine_tuning_schema(self, dataset):
        # Format error checks
        format_errors = defaultdict(int)

        for ex in dataset:
            if not isinstance(ex, dict):
                format_errors["data_type"] += 1
                continue

            # Check "messages" key
            messages = ex.get("messages", None)
            if not isinstance(messages, list):
                format_errors["missing_messages_list"] += 1
                continue

            if len(messages) == 0:
                format_errors["empty_messages_list"] += 1

            for message in messages:
                if not isinstance(message, dict):
                    format_errors["message_not_dict"] += 1
                    continue

                # Check that the role key is always present
                if "role" not in message:
                    format_errors["message_missing_role"] += 1

                # Check that either content or tool_calls must be present
                if "content" not in message and "tool_calls" not in message:
                    format_errors["message_missing_content_or_tool_calls"] += 1

                if any(k not in {"role", "content", "tool_calls"} for k in message):
                    format_errors["message_unrecognized_key"] += 1

                if message.get("role") not in {"system", "user", "assistant", "function"}:
                    format_errors["unrecognized_role"] += 1

                content = message.get("content", None)
                tool_calls = message.get("tool_calls", None)

                if (content is None and tool_calls is None) or (content and not isinstance(content, str)):
                    format_errors["missing_or_invalid_content"] += 1

                # Check tool_calls if present
                if tool_calls is not None:
                    if not isinstance(tool_calls, list):
                        format_errors["tool_calls_not_list"] += 1
                        continue

                    if len(tool_calls) == 0:
                        format_errors["empty_tool_calls_list"] += 1

                    for call in tool_calls:
                        if not isinstance(call, dict):
                            format_errors["tool_call_not_dict"] += 1
                            continue

                        required_tool_call_keys = {"id", "type", "function"}
                        if not required_tool_call_keys.issubset(call.keys()):
                            format_errors["tool_call_missing_key"] += 1
                            continue

                        if call.get("type") != "function":
                            format_errors["tool_call_invalid_type"] += 1

                        function = call.get("function", None)
                        if not function or not isinstance(function, dict):
                            format_errors["tool_call_invalid_function"] += 1
                            continue

                        required_function_keys = {"name", "arguments"}
                        if not required_function_keys.issubset(function.keys()):
                            format_errors["function_missing_key"] += 1

            # Check for at least one assistant message
            if not any(message.get("role") == "assistant" for message in messages):
                format_errors["example_missing_assistant_message"] += 1

            # Check "tools" key
            tools = ex.get("tools", None)
            if not isinstance(tools, list):
                format_errors["missing_tools_list"] += 1
                continue

            if len(tools) == 0:
                format_errors["empty_tools_list"] += 1

            for tool in tools:
                if not isinstance(tool, dict):
                    format_errors["tool_not_dict"] += 1
                    continue

                required_tool_keys = {"type", "function"}
                if not required_tool_keys.issubset(tool.keys()):
                    format_errors["tool_missing_key"] += 1

                if tool.get("type") != "function":
                    format_errors["tool_invalid_type"] += 1

                function = tool.get("function", None)
                if not function or not isinstance(function, dict):
                    format_errors["tool_invalid_function"] += 1
                    continue

                required_function_keys = {"name", "description", "parameters"}
                if not required_function_keys.issubset(function.keys()):
                    format_errors["function_missing_required_keys"] += 1

                parameters = function.get("parameters", None)
                if not parameters or not isinstance(parameters, dict):
                    format_errors["function_invalid_parameters"] += 1
                    continue

                required_parameters_keys = {"type", "properties", "required"}
                if not required_parameters_keys.issubset(parameters.keys()):
                    format_errors["parameters_missing_key"] += 1

                # Check "properties" within parameters
                properties = parameters.get("properties", None)
                if not isinstance(properties, dict):
                    format_errors["parameters_invalid_properties"] += 1
                    continue

                # Check each property within properties
                for prop_key, prop_value in properties.items():
                    if not isinstance(prop_value, dict):
                        format_errors["property_not_dict"] += 1
                        continue

                    if "type" not in prop_value:
                        format_errors["property_missing_type"] += 1

                    if "description" in prop_value and not isinstance(prop_value.get("description"), str):
                        format_errors["property_invalid_description"] += 1

                    if "enum" in prop_value:
                        if not isinstance(prop_value["enum"], list):
                            format_errors["property_invalid_enum"] += 1

        if format_errors:
            print("Found errors:")
            for k, v in format_errors.items():
                print(f"{k}: {v}")
        else:
            print("No errors found")

def check_files_in_directory(directory):
    schema_checker = OpenAIFineTuningSchemaCheck()

    # Iterate over each file in the directory
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)

        if os.path.isfile(file_path) and filename.endswith('.jsonl'):
            print(f"Checking file: {filename}")

            dataset = []
            with open(file_path, 'r', encoding='utf-8') as file:
                for line in file:
                    try:
                        # Parse each line as JSON and append to the dataset
                        json_object = json.loads(line)
                        dataset.append(json_object)
                    except json.JSONDecodeError as e:
                        print(f"Error decoding JSON in file {filename}: {e}")

            # Apply schema check to the dataset
            schema_checker.evaluate_open_ai_fine_tuning_schema(dataset)


if __name__ == "__main__":
    # Set up directory path
    load_dotenv()
    base_dir = Path(os.getenv('PROJECT_BASE_DIR'))
    text_files_dir = os.path.join(base_dir, "test_data", "fine_tuning")

    # Check files in the directory
    check_files_in_directory(text_files_dir)