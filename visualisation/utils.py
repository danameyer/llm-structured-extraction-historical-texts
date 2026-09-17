def format_model_name(model_dir: str) -> str:
    name = model_dir.removeprefix("model_")

    if name.startswith("openai_"):
        return name.removeprefix("openai_")

    if name.startswith("ollama_"):
        return name.removeprefix("ollama_")

    return name