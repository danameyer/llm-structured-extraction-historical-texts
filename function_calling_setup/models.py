from dataclasses import replace
from function_calling_setup.model_config import ModelConfig


OPENAI_LUNA = ModelConfig(
    provider="openai",
    model="gpt-5.6-luna",
    result_name="openai_gpt-5.6-luna",
    reasoning_effort="none"
)

OPENAI_TERRA = ModelConfig(
    provider="openai",
    model="gpt-5.6-terra",
    result_name="openai_gpt-5.6-terra",
    reasoning_effort="none"
)

OPENAI_SMOKE_TEST = ModelConfig(
    provider="openai",
    model="gpt-4o-2024-08-06",
    result_name="openai_gpt-4o-2024-08-06"
)

QWEN_36_35B = ModelConfig(
    provider="ollama",
    model="qwen3.6:35b",
    result_name="ollama_qwen3.6-35b",
    think=False
)

GPT_OSS_20B = ModelConfig(
    provider="ollama",
    model="gpt-oss:20b",
    result_name="ollama_gpt-oss-20b"
)

GEMMA4_26B = ModelConfig(
    provider="ollama",
    model="gemma4:26b",
    result_name="ollama_gemma4-26b",
    think=False
)

MISTRAL_SMALL_32_24B = ModelConfig(
    provider="ollama",
    model="mistral-small3.2:24b",
    result_name="ollama_mistral-small3.2-24b"
)

OPENAI_MODELS = [
    OPENAI_LUNA,
    OPENAI_TERRA
]

PROMPT_SELECTION_MODELS = [
    OPENAI_LUNA,
    # QWEN_36_35B  # add once OllamaProvider exists
]

MAIN_COMPARISON_MODELS = [
    OPENAI_LUNA,
    OPENAI_TERRA,
    # QWEN_36_35B,
    # GPT_OSS_20B,
    # GEMMA4_26B,
    # MISTRAL_SMALL_32_24B
]

REPEATABILITY_MODELS = [
    OPENAI_LUNA,
    # QWEN_36_35B
]

OPENAI_LUNA_REASONING_BASELINE = replace(
    OPENAI_LUNA,
    reasoning_effort="none"
)

OPENAI_LUNA_REASONING_ENABLED = replace(
    OPENAI_LUNA,
    reasoning_effort="medium"
)


QWEN_36_35B_REASONING_BASELINE = replace(
    QWEN_36_35B,
    think=False
)

QWEN_36_35B_REASONING_ENABLED = replace(
    QWEN_36_35B,
    think=True
)


GPT_OSS_20B_REASONING_BASELINE = replace(
    GPT_OSS_20B,
    think="low"
)

GPT_OSS_20B_REASONING_ENABLED = replace(
    GPT_OSS_20B,
    think="medium"
)


GEMMA4_26B_REASONING_BASELINE = replace(
    GEMMA4_26B,
    think=False,
)

GEMMA4_26B_REASONING_ENABLED = replace(
    GEMMA4_26B,
    think=True
)


REASONING_MODEL_PAIRS = [
    (
        OPENAI_LUNA_REASONING_BASELINE,
        OPENAI_LUNA_REASONING_ENABLED,
    ),
    # Uncomment once OllamaProvider is implemented:
    # (
    #     QWEN_36_35B_REASONING_BASELINE,
    #     QWEN_36_35B_REASONING_ENABLED,
    # ),
    # (
    #     GPT_OSS_20B_REASONING_BASELINE,
    #     GPT_OSS_20B_REASONING_ENABLED,
    # ),
    # (
    #     GEMMA4_26B_REASONING_BASELINE,
    #     GEMMA4_26B_REASONING_ENABLED,
    # ),
]