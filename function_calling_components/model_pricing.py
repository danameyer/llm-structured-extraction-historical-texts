class ModelPricing:

    def __init__(self):
        # Prices in USD per 1 million tokens.
        self.price_per_million_tokens = {
            "gpt-5.6-luna": {
                "input": 0.20,
                "output": 1.20,
            },
            "gpt-5.6-terra": {
                "input": 2.00,
                "output": 12.00,
            },
            "gpt-5.6-sol": {
                "input": 4.00,
                "output": 20.00,
            },

            # Keep temporarily for smoke tests.
            "gpt-4o-2024-08-06": {
                "input": 2.50,
                "output": 10.00,
            },
        }

    def get_price_per_token(self, model_name):
        pricing = self.price_per_million_tokens[model_name]

        return {
            "input": pricing["input"] / 1_000_000,
            "output": pricing["output"] / 1_000_000,
        }