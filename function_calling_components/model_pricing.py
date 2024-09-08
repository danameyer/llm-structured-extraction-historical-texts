class ModelPricing:
    def __init__(self):

        self.regular_divisor = 1000
        self.ft_divisor = 1000000

        # Price is per 1000 tokens
        self.regular_model_pricing = {
            "gpt-3.5-turbo": {
                "input": 0.0005,
                "output": 0.0015
            },
            "gpt-3.5-turbo-0125": {
                "input": 0.0005,
                "output": 0.0015
            },
            "gpt-4o": {
                "input": 0.005,
                "output": 0.015
            },
            "gpt-4o-mini": {
                "input": 0.00015,
                "output": 0.0006
            },
            "gpt-4-turbo": {
                "input": 0.01,
                "output": 0.03
            }
        }

        # Price is per 1M tokens
        self.ft_model_pricing = {
            "gpt-3.5-turbo-0125": {
                "input": 3.000,
                "output": 6.000,
                "training": 8.000
            },
            "gpt-4o-2024-08-06": {
                "input": 3.750,
                "output": 15.000,
                "training": 25.000
            },
            "gpt-4o-mini-2024-07-18": {
                "input": 0.300,
                "output": 1.200,
                "training": 3.000
            }
        }

    def get_price_per_token(self, model_name, is_finetune):
        if is_finetune:
            pricing = self.ft_model_pricing[model_name]
            pricing_per_token = {
                "intput": pricing["input"] / self.ft_divisor,
                "output": pricing["output"] / self.ft_divisor,
                "training": pricing["training"] / self.ft_divisor
            }
            return pricing_per_token
        else:
            pricing = self.regular_model_pricing[model_name]
            pricing_per_token = {
                "intput": pricing["input"] / self.regular_divisor,
                "output": pricing["output"] / self.regular_divisor
            }
            return pricing_per_token
