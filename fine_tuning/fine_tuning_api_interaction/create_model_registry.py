import json
import os


class ModelRegistry:
    def __init__(self, relative_path, registry_file_name):
        base_directory = os.getenv('PROJECT_BASE_DIR', '/default/path')
        if not base_directory:
            raise ValueError("PROJECT_BASE_DIR environment variable not set.")

        # relative_path = 'fine_tuning_files/openai/created_models'
        self.registry_file = os.path.join(base_directory, relative_path, registry_file_name)

        dir_name = os.path.dirname(self.registry_file)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name, exist_ok=True)

        self.registry = []
        self.load_registry()

    def load_registry(self):
        if os.path.exists(self.registry_file):
            with open(self.registry_file, 'r') as file:
                self.registry = json.load(file)
        else:
            self.registry = []

    def save_registry(self):
        with open(self.registry_file, 'w') as file:
            json.dump(self.registry, file, indent=4)

    def get_model_id(self, train_file_id):
        for entry in self.registry:
            if entry.get('train_file_id') == train_file_id:
                return entry['model_id']
        return None

    def add_model_id(self, train_file_id, validation_file_id, model_id, model_name):
        existing_entry = self.get_model_entry(train_file_id)
        if existing_entry:
            existing_entry['model_id'] = model_id
        else:
            self.registry.append({
                'train_file_id': train_file_id,
                'validation_file_id': validation_file_id,
                'model_id': model_id,
                'model_name': model_name
            })
        self.save_registry()

    def get_model_entry(self, train_file_id):
        for entry in self.registry:
            if entry.get('train_file_id') == train_file_id:
                return entry
        return None
