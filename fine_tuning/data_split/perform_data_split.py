import json
import os
import random
from pathlib import Path

from dotenv import load_dotenv


class FoldInfo:
    def __init__(self, fold_id):
        self.fold_id = fold_id
        self.training_set = list()
        self.validation_set = list()

    def add_training_file(self, training_file):
        self.training_set.append(training_file)

    def add_validation_file(self, validation_file):
        self.validation_set.append(validation_file)

    def to_dict(self):
        return self.__dict__


class PerformDataSplit:
    def __init__(self, jsonl_file, k, output_dir, model):
        """
        Initializes the PerformDataSplit instance.

        :param jsonl_file: Path to the input JSONL file.
        :param k: Number of folds for cross-validation.
        :param output_dir: Directory where the train and validation files will be saved.
        """
        self.jsonl_file = jsonl_file
        self.k = k
        self.output_dir = output_dir
        self.data = self._load_data()
        self.folds = self._split_into_folds()
        self.gpt_model = model

    def _load_data(self):
        """
        Loads data from the JSONL file.

        :return: List of JSON objects.
        """
        with open(self.jsonl_file, 'r') as file:
            data = [json.loads(line) for line in file]
        return data

    def _split_into_folds(self):
        """
        Splits the data into k folds.

        :return: List of k lists, each containing a subset of the data.
        """
        random.shuffle(self.data)
        fold_size = len(self.data) // self.k
        folds = [self.data[i * fold_size:(i + 1) * fold_size] for i in range(self.k)]

        # Handle any remaining data that doesn’t fit evenly into the folds
        remaining_data = self.data[self.k * fold_size:]
        for i in range(len(remaining_data)):
            folds[i % self.k].append(remaining_data[i])

        return folds

    def generate_train_val_sets(self):
        """
        Generates training and validation sets for each fold and writes them to files.
        """
        # Ensure the output directory exists
        os.makedirs(self.output_dir, exist_ok=True)

        fold_info_list: list[FoldInfo] = list()

        for i in range(self.k):
            # Create a directory for the current fold
            fold_dir = os.path.join(self.output_dir, f'fold_{i}')
            os.makedirs(fold_dir, exist_ok=True)

            # Validation set for the current fold
            validation_set = self.folds[i]

            # Training set is the concatenation of all other folds
            training_set = [item for j, fold in enumerate(self.folds) if j != i for item in fold]

            # Calculate the size of the validation set as 20% of the training set size
            training_size = len(training_set)
            validation_size = max(1, int(0.2 * training_size))  # Ensure at least one validation sample

            # Split training set to create validation set
            if validation_size < len(validation_set):
                validation_set = validation_set[:validation_size]
                training_set.extend(self.folds[i][validation_size:])  # Add remaining to training set
            else:
                validation_size = len(validation_set)  # Use all validation set if too small

            # Construct file paths for the current fold
            train_file_path = os.path.join(fold_dir, 'train.jsonl')
            val_file_path = os.path.join(fold_dir, 'val.jsonl')

            fold_info_file_path = os.path.join(fold_dir, 'fold_info_file.json')
            fold_info = FoldInfo(i)
            # fold_info = dict()
            # fold_info['fold'] = i
            # fold_info['training_set'] = list()
            # fold_info['validation_set'] = list()

            # Write the training and validation sets to files
            with open(train_file_path, 'w') as train_file, \
                    open(val_file_path, 'w') as val_file:
                for item in training_set:
                    train_file.write(json.dumps(item['jsonl']) + '\n')
                    fold_info.add_training_file(item['text_file'])
                for item in validation_set:
                    val_file.write(json.dumps(item['jsonl']) + '\n')
                    fold_info.add_validation_file(item['text_file'])

            fold_info_list.append(fold_info)

            with open(fold_info_file_path, 'w') as fold_info_file:
                json.dump(fold_info.to_dict(), fold_info_file)

        return fold_info_list


if __name__ == '__main__':
    load_dotenv()
    base_dir = Path(os.getenv('PROJECT_BASE_DIR'))

    # gpt
    gpt_model = 'gpt-3.5-turbo'
    fine_tuning_dir_open_ai = os.path.join(base_dir,
                                           'fine_tuning_files',
                                           "openai",
                                           'fine_tuning_data',
                                           'total_ft_data')
    fine_tuning_files_open_ai = os.path.join(fine_tuning_dir_open_ai, 'file-finetune-openai.jsonl')
    output_directory_open_ai = os.path.join(base_dir,
                                            'fine_tuning_files',
                                            "openai",
                                            'fine_tuning_data')
    data_splitter = PerformDataSplit(fine_tuning_files_open_ai, 3, output_directory_open_ai, gpt_model)
    data_splitter.generate_train_val_sets()

    # mistral
    mistral_model = 'open-mistral-7b'
    fine_tuning_dir_mistral = os.path.join(base_dir,
                                           'fine_tuning_files',
                                           'mistral',
                                           'fine_tuning_data',
                                           'total_ft_data')
    fine_tuning_files_mistral = os.path.join(fine_tuning_dir_mistral, 'file-finetune-mistral.jsonl')
    output_directory_mistral = os.path.join(base_dir,
                                            'fine_tuning_files',
                                            'mistral',
                                            'fine_tuning_data')
    data_splitter = PerformDataSplit(fine_tuning_files_mistral, 3, output_directory_mistral, mistral_model)
    data_splitter.generate_train_val_sets()
