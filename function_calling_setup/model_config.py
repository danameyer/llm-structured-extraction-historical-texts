from dataclasses import dataclass
from typing import Literal


ProviderName = Literal["openai", "ollama"]
ReasoningEffort = Literal["none", "low", "medium", "high"]
ThinkingLevel = bool | Literal["low", "medium", "high"]


@dataclass(frozen=True)
class ModelConfig:
    provider: ProviderName
    model: str
    result_name: str
    reasoning_effort: ReasoningEffort | None = None
    think: ThinkingLevel | None = None

    @property
    def result_dir(self) -> str:
        return f"model_{self.result_name}"