from typing import List, Dict

from dotenv import load_dotenv
import os
import openai
from langchain_community.chat_models import ChatOpenAI
from openai import OpenAI

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
selected_model = "gpt-3.5-turbo-0125"


class Message:
    def __init__(self, role, content):
        self.role = role
        self.content = content


class FunctionBuilder:
    def __init__(self, name, description, parameters):
        self.name = name
        self.description = description
        if parameters:
            self.parameters = parameters
        else:
            self.parameters = []


class Property:
    def __init__(self, property_type, description):
        self.type = property_type
        self.description = description


class Parameter:
    def __init__(self, parameter_type: str,
                 properties: Dict[str, Property] = None,
                 required: List[str] = None):

        self.type = parameter_type

        if properties:
            self.properties: Dict[str, Property] = properties
        else:
            self.properties: Dict[str, Property] = dict()

        if required:
            self.required = required
        else:
            self.required = list()

    def add_property(self, name, property_type, description, required):
        if required:
            self.required.append(name)

        self.properties[name] = Property(property_type, description)


class ChatCompletion:

    def __init__(self, model):
        load_dotenv()
        self.client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
        )
        self.model = model

    def generate_response(self, messages, functions, function_call="auto"):
        """
        Generate response using OpenAI Chat Completion API.

        Args:
            self: self parameter
            messages (list): List of message dictionaries.
            functions (list, optional): List of function dictionaries. Defaults to None.
            function_call (str, optional): Type of function call to use. Defaults to "auto".

        Returns:
            str: Generated response.
        """
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                functions=functions,
                function_call=function_call
                # response_format={"type": "json_object"}
            )
            return completion.choices[0].message
        except openai.APIConnectionError as e:
            print("The server could not be reached")
            print(e.__cause__)
        except openai.RateLimitError as e:
            print("Rate limit has been exceeded.")
            print(f"Exception: {e}")
        except openai.APIStatusError as e:
            print("Another non-200-range status code was received")
            print(e.status_code)
            print(e.response)
        except openai.OpenAIError as e:
            print("An unexpected API error occurred.")
            print(f"Exception: {e}")
