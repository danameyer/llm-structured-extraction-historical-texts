from function_calling_setup.model_config import ModelConfig
from function_calling_setup.providers.base_provider import LLMProvider
from function_calling_setup.providers.openai_provider import OpenAIProvider


def create_provider(config: ModelConfig, strict: bool = False) -> LLMProvider:
    if config.provider == "openai":
        return OpenAIProvider(config=config, strict=strict)

    if config.provider == "ollama":
        raise NotImplementedError("OllamaProvider has not been implemented yet.")
        # return OllamaProvider(
        #     config=config,
        # )

    raise ValueError(f"Unsupported provider: {config.provider}")