class PromptBuilder:
    def __init__(self):
        self.prompting_strategies = []

    def concatenate_files(self, file_paths):
        concatenated_content = ""
        for file_path in file_paths:
            try:
                with open(file_path, 'r') as file:
                    content = file.read()
                    concatenated_content += f"'''{content}'''\n"
            except FileNotFoundError:
                print(f"File not found: {file_path}")
            except IOError:
                print(f"Error reading file: {file_path}")
        return concatenated_content

    def add_base_prompt(self, file_paths):
        prompt = (
            "Work on the following three tasks consecutively for each \"court document\" individually:\n"
            "(1) Extract information about the persons mentioned in the documents and their relations to each other.\n"
            "(2) Transfer your results to structured JSON output separately for each document by using a tool.\n"
            # "(3) Answer questions on the basis of the structured JSON output. Wait for the user to prompt you for questions.\n\n"
            "Individual \"court documents\" are contained within triple quotes. \"Court documents\":\n" +
            self.concatenate_files(file_paths)
        )
        self.prompting_strategies.append(prompt)
        return self

    def add_task_1(self):
        prompt = (
            "(1) Extract information about the persons mentioned in the documents and their relations to each other.\n"
        )
        self.prompting_strategies.append(prompt)
        return self

    def add_task_2(self):
        prompt = (
            "(2) Transfer your results to structured JSON output separately for each document by using a tool.\n"
        )
        self.prompting_strategies.append(prompt)
        return self

    def add_task_3(self):
        prompt = (
            "(3) Answer questions on the basis of the structured JSON output. Wait for the user to prompt you for questions.\n"
        )
        self.prompting_strategies.append(prompt)
        return self

    def add_input_text(self, file_paths):
        prompt = (
            "I will provide you with three tasks. Please work on each task consecutively for each \"court document\" individually.\n"
            "Individual \"court documents\" are contained within triple quotes. \"Court documents\":\n" +
            self.concatenate_files(file_paths)
        )
        self.prompting_strategies.append(prompt)
        return self

    def add_persona_modelling(self):
        prompt = (
            "You are an expert in the generation of structured JSON output from unstructured data and in the analysis of English historical legal documents in Middle Latin.\n"
            "Your target audience is historians well-versed in Medieval English history.\n"
        )
        self.prompting_strategies.append(prompt)
        return self

    def add_context(self):
        prompt = (
            "You will be presented with documents from court rolls from the 13th and 14th centuries which summarise court cases heard at courts in Medieval England. These documents list all the persons involved in the court cases and contain information about the locations they come from, their family relations, their professions and their power relations with regard to other persons mentioned in the document. The main language of the documents is medieval Latin but entities such as names, locations and professions may be in Middle English.\n"
        )
        self.prompting_strategies.append(prompt)
        return self

    def add_iterative_approach(self):
        prompt = (
            "Take your time to read through the following instructions carefully.\n"
            "Take a deep breath and take your time to work on the tasks step-by-step.\n"
        )
        self.prompting_strategies.append(prompt)
        return self

    def add_q_and_a_prompting(self):
        prompt = (
            "Ask questions if anything is unclear for a given step.\n"
        )
        self.prompting_strategies.append(prompt)
        return self

    def add_schema_information(self):
        prompt = (
            "This is the JSON schema:\n"
            "{\n"
            "  \"person_list\":\n"
            "  [\n"
            "    {\n"
            "      \"id\": ... ,\n"
            "      \"name\": \"...\",\n"
            "      \"cognomen\": \"...\",\n"
            "      \"profession\": \"...\",\n"
            "      \"family_relations\": [\n"
            "        {\n"
            "          \"relation_type\": \"...\",\n"
            "          \"related_person\": ...\n"
            "        }\n"
            "      ],\n"
            "      \"legal_relationship\": [\n"
            "        {\n"
            "          \"relation_type\": \"...\",\n"
            "          \"related_person\": ...\n"
            "        }\n"
            "      ],\n"
            "      \"place_of_origin\": \"...\",\n"
            "      \"title\": \"\"\n"
            "    }\n"
            "  ]\n"
            "}\n\n"
            "These are explanations of the categories in the JSON schema:\n"
            "* 'id': the unique id of the person\n"
            "* 'name': the name of the person\n"
            "* 'cognomen': additional nickname of a person\n"
            "* 'place of origin': the place the person comes from\n"
            "* 'profession': the work the person does\n"
            "* 'family relations' with 'relation type' (the role the named person takes in the relationship) and 'related person' (the id of the person the named person is related to)\n"
            "* 'legal relationship' with 'relation type' (type of permanent legal social power relationship such as custos and heres) and 'related person' (the id of the person the named person is related to in a legal sense)\n"
            "* 'title': the official title of the person (i.e. within the Church or nobility)\n"
        )
        self.prompting_strategies.append(prompt)
        return self

    def add_constraints(self):
        prompt = (
            "While working on the tasks, pay attention to the following rules:\n"
            "Rules:\n"
            "* Leave values empty if there is no information provided in the text.\n"
            "* Remember to include the id.\n"
            "* Use the nominative singular form.\n"
            "* Stick to the spelling variations used in the document.\n"
            "* If word endings are cut due to OCR errors, reconstruct the complete word.\n"
            "* Use Latin terms for all of the values in the JSON file.\n"
            "* Put values preceded by the preposition 'de' into the 'place_of_origin' category if they refer to English place names.\n"
            "* Professions often follow the name and are often preceded by 'le'. Assign professions to the 'profession' category.\n"
            "* Pay attention to the use of pronouns or attributes such as 'predictus' in the text to make out persons already mentioned before.\n"
            "* List persons with the same names who are different persons.\n"
            "* Only include the first name in the name category.\n"
            "* Values in the 'cognomen' category keep the prepositions 'le' and 'de' if they are preceded by them in the text.\n"
            "* Jobs in the Church are assigned to the 'title' category.\n"
            "* List 'profession' without the preposition 'le'.\n"
            "* List 'place_of_origin' without the preposition 'de'.\n"
            "* French location names usually refer to names of the nobility and belong to the 'cognomen' category and are listed with 'de'.\n"
            "* The information after the name of a person is assigned to the 'cognomen' category.\n"
            "* Relationship_types must describe the person listed (i.e. a woman is 'Uxor' and a man 'Maritus').\n"
            "* Titles like Rex or Prior are listed only as 'title'.\n"
            "* Spell all the values you write into the JSON file with a capital letter at the beginning of a word.\n"
            "* For power relations, the following relations exist:\n"
            "  ** 'Tenens' (tenant) vs. 'Dominus feodi' (feudal lord)\n"
            "  ** 'Testator' (person inherited from) vs. 'Heres' (heir)\n"
            "  ** 'Plegiarius' (person who grants surety), one-sided relationship assigned to the person who is the subject of the relationship.\n"
            "  ** 'Attornatus' (attorney), one-sided relationship assigned to the attorney in a legal context.\n"
            "  ** 'Reus' (defendant) vs. 'Petitor' (plaintiff) ('Reus' can also be assigned without an id link to another person if the plaintiff is not listed).\n"
            "  ** 'Custos' (warden) vs. 'Pupillus' (the ward).\n"
            "  ** 'Progenitor' (ancestor) vs 'Progenies' (descendant).\n"
        )
        self.prompting_strategies.append(prompt)
        return self

    def add_emotional_prompting(self):
        prompt = (
            "It is very important for my research that you deliver accurate results!\n"
            "I'll tip you $200 dollars for the best answer!\n"
        )
        self.prompting_strategies.append(prompt)
        return self

    def add_demonstrations(self, demonstrations):
        prompt = (
            "Examples are contained in triple quotes.\n"
            "Consider these examples only for orientation in your work on the \"court documents\":\n"
            "**Examples**\n" +
            self.concatenate_files(demonstrations)
        )
        self.prompting_strategies.append(prompt)
        return self

    def build_prompt(self):
        return '\n\n'.join(self.prompting_strategies).strip()
