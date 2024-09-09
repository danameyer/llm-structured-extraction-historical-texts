import os


class ProgressCheck:
    def __init__(self, dir_for_fold_data: str, dir_for_text_data_copy: str | None, number_of_folds: int):
        self._dir_for_fold_data = dir_for_fold_data
        self._dir_for_text_data_copy = dir_for_text_data_copy
        self._number_of_folds = number_of_folds
        self._create_message_object_performed = False
        self._data_split_performed = False
        self._copy_text_files_performed = False
        self._training_performed = False

    def tick_creation_of_message_object_performed(self):
        self._create_message_object_performed = True

    def tick_data_split_performed(self):
        self._data_split_performed = True

    def tick_copy_text_files_performed(self):
        self._copy_text_files_performed = True

    def tick_training_performed(self):
        self._training_performed = True

    def is_training_allowed(self):
        if self._data_split_performed and self._copy_text_files_performed:
            return True
        elif ((not self._data_split_performed)
              and (not self._copy_text_files_performed)
              and self._does_fold_data_exist()
              and self._do_text_file_copies_exist()):
            return True
        else:
            return False

    def is_training_allowed_for_mistral(self):
        if self._data_split_performed:
            return True
        elif ((not self._data_split_performed)
              and self._does_fold_data_exist()):
            return True
        else:
            return False

    def _does_fold_data_exist(self):
        content_fold_data_dir = os.listdir(self._dir_for_fold_data)
        fold_dirs = [fold_dir for fold_dir in content_fold_data_dir if fold_dir.startswith("fold_")]

        if len(fold_dirs) != self._number_of_folds:
            return False

        # Check files in detail
        fold_info_file_json = "fold_info_file.json"
        train_jsonl = "train.jsonl"
        val_jsonl = "val.jsonl"
        fold_paths = [os.path.join(self._dir_for_fold_data, fold_dir) for fold_dir in fold_dirs]
        for fold_path in fold_paths:
            fold_contents = os.listdir(fold_path)
            if fold_info_file_json not in fold_contents:
                print(f"Missing file in {fold_path}: {fold_info_file_json}")
                return False
            if train_jsonl not in fold_contents:
                print(f"Missing file in {fold_path}: {train_jsonl}")
                return False
            if val_jsonl not in fold_contents:
                print(f"Missing file in {fold_path}: {val_jsonl}")
                return False

        return True

    def _do_text_file_copies_exist(self):

        if self._dir_for_text_data_copy is None:
            raise FileNotFoundError("Directory with copied text files was not defined.")

        content_text_files_dir = os.listdir(self._dir_for_text_data_copy)
        fold_dirs = [fold_dir for fold_dir in content_text_files_dir if fold_dir.startswith("fold_")]

        if len(fold_dirs) != self._number_of_folds:
            return False

        # Check files in detail
        # training = "training_text_files"
        validation = "validation_text_files"
        fold_paths = [os.path.join(self._dir_for_text_data_copy, fold_dir) for fold_dir in fold_dirs]
        for fold_path in fold_paths:
            fold_contents = os.listdir(fold_path)
            # if training not in fold_contents:
            #     print(f"Missing directory in {fold_path}: {training}")
            #     return False
            if validation not in fold_contents:
                print(f"Missing directory in {fold_path}: {validation}")
                return False

            # training_path = os.path.join(fold_path, training)
            validation_path = os.path.join(fold_path, validation)
            # training_contents = os.listdir(training_path)
            validation_contents = os.listdir(validation_path)

            if len(validation_contents) == 0:
                return False

        return True

