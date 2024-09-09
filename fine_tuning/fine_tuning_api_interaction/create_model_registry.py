import json
import os
import datetime


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

    # def get_model_id(self, train_file_id):
    #     for entry in self.registry:
    #         if entry.get('train_file_id') == train_file_id:
    #             return entry['model_id']
    #     return None

    def add_model_entry(self,
                        train_file_id,
                        validation_file_id,
                        ft_job_id,
                        ft_model_id,
                        model_name,
                        fold_number,
                        token_number,
                        epoch_number,
                        duration,
                        training_price):
        # existing_entry = self.get_model_entry(train_file_id)
        # if not existing_entry['ft_model_id'] == ft_model_id:
        self.registry.append({
            'time_stamp': str(datetime.datetime.now()),
            'train_file_id': train_file_id,
            'validation_file_id': validation_file_id,
            'ft_job_id': ft_job_id,
            'ft_model_id': ft_model_id,
            'model_name': model_name,
            'fold_number': fold_number,
            'token_number': token_number,
            'epoch_number': epoch_number,
            'duration': duration,
            'training_price': training_price
        })
        self.save_registry()

    # def get_model_entry(self, ft_model_id):
    #     for entry in self.registry:
    #         if entry.get('ft_model_id') == ft_model_id:
    #             return entry
    #     return None
