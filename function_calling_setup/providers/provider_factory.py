from function_calling_setup.models.model_config import ModelConfig
from function_calling_setup.providers.base_provider import LLMProvider
from function_calling_setup.providers.ollama_provider import OllamaProvider
from function_calling_setup.providers.openai_provider import OpenAIProvider


def create_provider(config: ModelConfig, strict: bool = False) -> LLMProvider:
    if config.provider == "openai":
        return OpenAIProvider(config=config, strict=strict)

    if config.provider == "ollama":
        return OllamaProvider(config=config)

    raise ValueError(f"Unsupported provider: {config.provider}")