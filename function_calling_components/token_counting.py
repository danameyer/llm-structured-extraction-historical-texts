import tiktoken
from typing import List


class TokenCounter:
    def __init__(self):
        pass

    @staticmethod
    def count_input_tokens(messages: List[dict], model: str) -> int:
        encoding = tiktoken.encoding_for_model(model)
        total_tokens = 0
        for message in messages:
            content = message.get('content', '')
            tokens = encoding.encode(content)
            total_tokens += len(tokens)
        return total_tokens

    @staticmethod
    def count_output_tokens(response: str, model: str) -> int:
        if not isinstance(response, str):
            response = str(response)

        encoding = tiktoken.encoding_for_model(model)
        tokens = encoding.encode(response)
        return len(tokens)

    @staticmethod
    def calculate_costs(input_token_count: int, output_token_count: int, model: str) -> float:
        model_pricing = {
            "gpt-3.5-turbo": {
                "input": 0.005,
                "output": 0.015
            },
            "gpt-3.5-turbo-0125": {
                "input": 0.005,
                "output": 0.015
            },
            "gpt-4o": {
                "input": 0.005,
                "output": 0.015
            }
        }
        input_cost_per_1000_tokens = model_pricing.get(model, {}).get("input", 0)
        output_cost_per_1000_tokens = model_pricing.get(model, {}).get("output", 0)

        input_cost = (input_token_count / 1000) * input_cost_per_1000_tokens
        output_cost = (output_token_count / 1000) * output_cost_per_1000_tokens

        total_cost = input_cost + output_cost
        return total_cost

