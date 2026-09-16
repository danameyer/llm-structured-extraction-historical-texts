import tiktoken
from typing import List
from function_calling_setup.model_pricing import ModelPricing


class TokenCounter:
    def __init__(self, model):
        self.model = model
        self.input_tokens = 0
        self.output_tokens = 0

    def add_number_of_input_tokens(self, number_of_input_tokens):
        self.input_tokens += number_of_input_tokens

    def add_number_of_output_tokens(self, number_of_output_tokens):
        self.output_tokens += number_of_output_tokens

    def add_input(self, input_message):
        self.input_tokens += self._count_input_tokens(input_message)

    def add_output(self, output_message):
        self.output_tokens += self._count_output_tokens(output_message)

    def get_costs(self):
        return self._calculate_costs(self.input_tokens, self.output_tokens)

    def get_cost_summary(self, pred_filename):
        info = {
            "filename": pred_filename,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "costs": self.get_costs()
        }
        return info

    def _count_input_tokens(self, messages: List[dict]) -> int:
        encoding = tiktoken.encoding_for_model(self.model)
        total_tokens = 0
        for message in messages:
            content = message.get('content', '')
            tokens = encoding.encode(content)
            total_tokens += len(tokens)
        return total_tokens

    def _count_output_tokens(self, response: str) -> int:
        if not isinstance(response, str):
            response = str(response)

        encoding = tiktoken.encoding_for_model(self.model)
        tokens = encoding.encode(response)
        return len(tokens)

    def _calculate_costs(self, input_token_count: int, output_token_count: int) -> float | None:

        model_pricing = ModelPricing()
        pricing_per_token = model_pricing.get_price_per_token(model_name=self.model)

        if pricing_per_token is None:
            return None

        input_cost = pricing_per_token["input"] * input_token_count
        output_cost = pricing_per_token["output"] * output_token_count

        return input_cost + output_cost

