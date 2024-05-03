import json
import os

import jsons
from openai import OpenAI
from dotenv import load_dotenv

from chat_completion_blueprint import Parameter, FunctionBuilder, Property, ChatCompletion, Message


def main():
    load_dotenv()
    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
    )

    # # Hello World
    # result = client.chat.completions.create(
    #     messages=[
    #         {
    #             "role": "user",
    #             "content": "Tell me a joke about cats.",
    #         }
    #     ],
    #     model="gpt-3.5-turbo",
    # )
    #
    # print(result.to_json())
    #
    # # hello world info extraction
    # student_1_description = "David Nguyen is a sophomore majoring in computer science at Stanford University. He is Asian American and has a 3.8 GPA. David is known for his programming skills and is an active member of the university's Robotics Club. He hopes to pursue a career in artificial intelligence after graduating."
    # # A simple prompt to extract information from "student_description" in a JSON format.
    # prompt_1 = f'''
    # Please extract the following information from the given text and return it as a JSON object:
    #
    # name
    # major
    # school
    # grades
    # club
    #
    # This is the body of text to extract the information from:
    # {student_1_description}
    # '''
    #
    # openai_response = client.chat.completions.create(
    #     model='gpt-3.5-turbo',
    #     messages=[{'role': 'user', 'content': prompt_1}]
    # )
    #
    # print(openai_response.choices[0].message.content)

    # hello world function calling
    # Define the function extract_entities
    extract_entities_function = FunctionBuilder(
        name="extract_entities",
        description="Extract named entities from text",
        parameters=Parameter(
            parameter_type="object",
            properties={
                "person": Property(property_type="string", description="name of a person"),
                "location": Property(property_type="string", description="name of a location")
            },
            required=["person", "location"]
        )
    )
    extract_entities_as_dict = json.loads(jsons.dumps(extract_entities_function))

    message = Message(role="user",
                      content="Can you extract named entities from 'John Doe is a software engineer at Google. He lives in New York City.'?")

    message_as_dict = json.loads(jsons.dumps(message))

    # Create an instance of ChatCompletion
    chat_completion = ChatCompletion(model='gpt-3.5-turbo')

    # Generate response with a user message and the extract_entities function
    response = chat_completion.generate_response(
        messages=[message_as_dict],
        functions=[extract_entities_as_dict]
    )

    print(response)

    # function_descriptions = [
    #     {
    #         "name": "get_flight_info",
    #         "description": "Get flight information between two locations",
    #         "parameters": {
    #             "type": "object",
    #             "properties": {
    #                 "loc_origin": {
    #                     "type": "string",
    #                     "description": "The departure airport, e.g. DUS",
    #                 },
    #                 "loc_destination": {
    #                     "type": "string",
    #                     "description": "The destination airport, e.g. HAM",
    #                 },
    #             },
    #             "required": ["loc_origin", "loc_destination"],
    #         },
    #     }
    # ]
    #
    # user_prompt = "When's the next flight from Amsterdam to New York?"
    #
    # completion = client.chat.completions.create(
    #     model="gpt-3.5-turbo-0613",
    #     messages=[{"role": "user", "content": user_prompt}],
    #     # Add function calling
    #     functions=function_descriptions,
    #     function_call="auto",  # specify the function call
    # )
    #
    # # It automatically fills the arguments with correct info based on the prompt
    # # Note: the function does not exist yet
    #
    # output = completion.choices[0].message
    # print(output)


if __name__ == '__main__':
    main()
