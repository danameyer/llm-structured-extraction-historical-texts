import json
import os

from dotenv import load_dotenv


class CountOfPersonsInJson:
    def __init__(self):
        pass

    @staticmethod
    def count_number_of_persons_in_json(gt_folder):

        total_person_count = 0

        for filename in os.listdir(gt_folder):
            if filename.endswith(".json"):
                file_path = os.path.join(gt_folder, filename)

                with open(file_path, 'r', encoding='utf-8') as json_file:
                    data = json.load(json_file)

                    if 'person_list' in data:
                        person_count = len(data['person_list'])
                        print(f"{filename} contains {person_count} person entries.")
                        total_person_count += person_count

        return total_person_count

    @staticmethod
    def count_words_in_texts(txt_folder):
        text_length_dict = {}

        for file_name in os.listdir(txt_folder):
            if file_name.endswith('.txt'):
                file_path = os.path.join(txt_folder, file_name)

                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()

                    word_count = len(content.split())

                    text_length_dict[file_name] = word_count

        return text_length_dict

    @staticmethod
    def return_min_max_text_length(text_length_dict):
        min_file = min(text_length_dict, key=text_length_dict.get)
        max_file = max(text_length_dict, key=text_length_dict.get)

        return {
            'min_file': (min_file, text_length_dict[min_file]),
            'max_file': (max_file, text_length_dict[max_file])
        }


if __name__ == '__main__':
    load_dotenv()
    base_dir = os.getenv('PROJECT_BASE_DIR')
    gt_folder_path = os.path.join(base_dir, "data", "test_gt")
    personCounter = CountOfPersonsInJson()
    number_of_persons = personCounter.count_number_of_persons_in_json(gt_folder_path)
    print("This is the total number of persons in the documents:", number_of_persons)

    txt_folder_path = os.path.join(base_dir, "data", "test_txt")
    dict_text_length = personCounter.count_words_in_texts(txt_folder_path)
    min_max_text_length = personCounter.return_min_max_text_length(dict_text_length)
    print(min_max_text_length)
