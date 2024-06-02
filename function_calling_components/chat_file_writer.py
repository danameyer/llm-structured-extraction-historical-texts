import os
from datetime import datetime
from pathlib import Path


class ChatFileWriter:
    def __init__(self):
        self._timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        base_dir = Path(os.getenv('PROJECT_BASE_DIR'))
        self._prompt_save_dir = os.path.join(base_dir, "prompting", "generated_prompts")
        self._response_save_dir = os.path.join(base_dir, "prompting", "prompting_responses")

    def _save_to_file(self, out_path, content, filename):
        filename_with_timestamp = f"{filename}_{self._timestamp}.txt"
        file_path = os.path.join(out_path, filename_with_timestamp)
        with open(file_path, 'a') as file:
            file.write(content + '\n')

    def save_response(self, content, filename):
        self._save_to_file(self._response_save_dir, content, filename)

    def save_prompt(self, content, filename):
        self._save_to_file(self._prompt_save_dir, content, filename)

