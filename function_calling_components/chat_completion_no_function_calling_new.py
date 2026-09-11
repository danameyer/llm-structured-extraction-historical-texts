import json
from typing import List, Union

import jsons
import openai
from jsons import ValidationError
from openai.types.chat import ChatCompletion
from tenacity import stop_after_attempt, wait_random_exponential, retry

from evaluation.json_comparison.json_validation import JsonValidator
from function_calling_components.chat_completion_blueprint import DialogueCompletion
from function_calling_components.function_calling import Message, SimpleChatGptMessage
from functions.BaseFunction import BaseFunction
from functions.ExtractJsonFromPlainText import ExtractJsonFromPlainText


class DialogueCompletionNoFunctionCallingNew(DialogueCompletion):

    def __init__(self, model, experiment_dir):
        super().__init__(model, experiment_dir)
        self.json_result = None

    def prompt_assistant_response(self,
                                  prompt,
                                  filename,
                                  function_list=None,
                                  print_conversation=True,
                                  validate=True,
                                  require_json_output=False):
        self._append_message(Message("user", prompt))

        self.chat_file_writer.save_prompt(prompt, filename)

        try:
            chat_response = self._execute_chat_completion_query(
                messages=self.message_history,
                tools=function_list,
                validate=validate,
                require_json_output=require_json_output
            )
            assistant_message = chat_response.choices[0].message.content

            self._append_message(Message("assistant", assistant_message))
            self.chat_file_writer.save_response(assistant_message, filename)

            if print_conversation:
                self._print_conversation()

            return assistant_message

        except Exception as ex:
            if require_json_output:
                print(f"Error while requesting JSON output: {ex}")

    @retry(stop=stop_after_attempt(6), wait=wait_random_exponential(multiplier=1, max=10), reraise=True, )
    def _execute_chat_completion_query(self,
                                       messages: List[Message],
                                       tools: List[BaseFunction] = None,
                                       validate=True,
                                       require_json_output=False) -> Union[ChatCompletion, None]:

        completion = self._request_response(messages=messages,
                                            functions=tools,
                                            require_json_output=require_json_output)

        if require_json_output:
            print("JSON output required - clean it up ...")
            function_call_result = self._clean_up_json_output(completion, messages, tools, validate)
            return function_call_result
        else:
            print("No function called.")
            return completion

    def _request_response(self,
                          messages: List[Message],
                          functions: List[BaseFunction] = None,
                          function_call="auto",
                          require_json_output: bool = False) -> Union[ChatCompletion, None]:
        """
        Generate response using OpenAI Chat Completion API.

        Args:
            self: self parameter
            messages (list): List of message objects.
            functions (list, optional): List of function dictionaries. Defaults to None.
            function_call (str, optional): Type of function call to use. Defaults to "auto".

        Returns:
            Completion: Generated response.
        """
        return self._request_response_without_function_call(messages)

    def _request_response_without_function_call(self, messages: List[Message], require_json_output: bool = False):
        try:
            simple_chat_gpt_messages = [SimpleChatGptMessage(role=x.role, content=x.content, name=x.name) for x in messages]
            messages_as_dict = json.loads(jsons.dumps(simple_chat_gpt_messages))
            print(f"Input response content: {messages_as_dict}")
            # functions_as_dict_list = [x.get_definition_dict() for x in functions] if functions else None

            self.runtime_calculator.start()

            if require_json_output:
                function = ExtractJsonFromPlainText()
                function_dict = function.get_definition_dict()
                schema = function_dict["function"]["parameters"]
                response_format = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "person_list_schema",
                        "strict": json.loads(json.dumps(True)),
                        "schema": schema
                    }
                }
                completion: Union[ChatCompletion, None] = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages_as_dict,
                    response_format=response_format
                )

            else:
                completion: Union[ChatCompletion, None] = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages_as_dict,
                )

            self.runtime_calculator.end()

            # Print the runtime
            print(f"API call duration: {self.runtime_calculator.calculate_runtime():.2f} seconds")

            output_response = completion.choices[0].message.content
            print(f"Output response content: {output_response}")

            self.token_counter.add_number_of_input_tokens(completion.usage.prompt_tokens)
            self.token_counter.add_number_of_output_tokens(completion.usage.completion_tokens)

            # function definitions have been sent to chatGPT - now only use the short description in order to
            # save tokens ...
            # self.flag_function_calls_for_short_description(functions)
            return completion
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

    def _clean_up_json_output(self,
                              completion: Union[ChatCompletion, None],
                              messages: List[Message],
                              tools: List[BaseFunction],
                              validate=True) -> Union[ChatCompletion, None]:
        # function_name = completion.choices[0].message.tool_calls[0].function.name
        content = completion.choices[0].message.content
        print(f"This content will be cleaned up: {content}")
        content_without_delimiters = content.replace("```json", "").replace("```", "")  # remove leading and trailing delimiter

        json_output_as_dict = json.loads(content_without_delimiters)
        print("This is the json_output as dict (it should look like function call arguments):", json_output_as_dict)

        # function_object = next((x for x in tools if x.get_definition().function.name == function_name), None)

        function_object = ExtractJsonFromPlainText()
        if function_object:
            try:
                self.json_result = function_object.run(**json_output_as_dict)
                print("This is the function call result", self.json_result)
            except Exception as ex:
                print("Could not execute function object: ", ex)
                raise ex

            try:
                json_validator = JsonValidator()
                if validate:
                    validation = json_validator.validate_json(self.json_result)
                    if not validation[0]:
                        message = "JSON validation failed: " + validation[1]
                        raise ValidationError(message)
            except Exception as ex:
                print("Could not validate json output.", ex)
                raise ex

            try:
                messages.append(
                    Message(role="function",
                            content=str(self.json_result),
                            name=function_object.get_definition().function.name,
                            # function_call_id=completion.choices[0].message.tool_calls[0].id,
                            # function_call_arguments=completion.choices[0].message.tool_calls[0].function.arguments
                            function_call_arguments=function_object.get_definition().function.parameters
                            )
                )
            except Exception as ex:
                print("Could not append message object: ", ex)
                raise ex

            try:
                response = self._request_response(messages)
                return response
            except Exception as e:
                print(type(e))
                raise Exception("Chat response could not be generated.")

