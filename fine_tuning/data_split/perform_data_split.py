import json
import os
import random
import shutil
from pathlib import Path
from typing import List

from dotenv import load_dotenv


class FoldInfo:
    def __init__(self, fold_id):
        self.fold_id: int = fold_id
        self.training_set: List[str] = list()
        self.validation_set: List[str] = list()

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

    def copy_text_files(self, fold_info_list: List[FoldInfo], path_to_text_files: str, output_directory: str):
        fold_directories = os.listdir(output_directory)
        if fold_directories:
            fold_dir_paths = [os.path.join(output_directory, fold_dir) for fold_dir in fold_directories]
            raise FileExistsError(f"""
                                One or more fold directories already exists within the output directory.
                                The fold directories should not exist. Please delete them. 
                                Found fold directories: {fold_dir_paths}
                                """)

        for fold_info in fold_info_list:
            self._copy_text_files_for_fold(fold_info, path_to_text_files, output_directory)

    def _copy_text_files_for_fold(self, fold_info: FoldInfo, path_to_text_files: str, output_directory: str):
        text_file_names_training = fold_info.training_set
        text_file_names_validation = fold_info.validation_set
        text_files_paths_training = [os.path.join(path_to_text_files, text_file_name) for text_file_name in text_file_names_training]
        text_files_paths_validation = [os.path.join(path_to_text_files, text_file_name) for text_file_name in text_file_names_validation]

        fold_id = fold_info.fold_id
        fold_dir = os.path.join(output_directory, f"fold_{fold_id}")
        training_dir = os.path.join(fold_dir, "training_text_files")
        validation_dir = os.path.join(fold_dir, "validation_text_files")
        os.makedirs(output_directory, exist_ok=True)

        # require empty output directory
        try:
            os.makedirs(fold_dir)
            os.makedirs(training_dir)
            os.makedirs(validation_dir)
        except OSError as error:
            raise FileExistsError(f"""
            The fold directory already exists but should not. Please delete it. 
            See error:
            {str(error)}
            """)

        copied_training_files: List[str] = list()
        for training_file_path in text_files_paths_training:
            destination = shutil.copy(training_file_path, training_dir)
            copied_training_files.append(destination)

        copied_validation_files: List[str] = list()
        for validation_file_path in text_files_paths_validation:
            destination = shutil.copy(validation_file_path, validation_dir)
            copied_validation_files.append(destination)

        return {
            "training_file_destinations": copied_training_files,
            "validation_file_destinations": copied_validation_files
        }

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

    data_dir = os.path.join(base_dir, 'data')
    test_gt_dir = os.path.join(data_dir, 'test_gt')
    test_txt_dir = os.path.join(data_dir, 'test_txt')

    training_evaluation_data_dir = os.path.join(base_dir,
                                                'evaluation_results',
                                                'txt_files_fine_tuning_evaluation_data')

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
    fold_info_list_openai = data_splitter.generate_train_val_sets()
    data_splitter.copy_text_files(fold_info_list_openai, test_txt_dir, training_evaluation_data_dir)

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
    fold_info_list_mistral = data_splitter.generate_train_val_sets()
