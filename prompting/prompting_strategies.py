class PromptBuilder:
    def __init__(self, file_paths):
        base_prompt = self.add_base_prompt(file_paths)
        self.prompting_strategies = []
        self.prompting_strategies.append(base_prompt)

    def concatenate_files(self, file_paths):
        concatenated_content = ""
        for file_path in file_paths:
            try:
                with open(file_path, 'r') as file:
                    content = file.read()
                    concatenated_content += f"'''{content}'''\n\n"
            except FileNotFoundError:
                print(f"File not found: {file_path}")
            except IOError:
                print(f"Error reading file: {file_path}")
        return concatenated_content

    def add_base_prompt(self, file_paths):
        base_prompt = """Work on the following three tasks consecutively for each document individually:
        (1) Extract information about the persons mentioned in the documents and their relations 
        to each other. 
        (2) Transfer your results to structured JSON output separately for each document. 
        (3) Answer questions on the basis of the structured JSON output. Wait for the user to prompt 
        you for questions.
        
        Individual documents are contained within triple 
        quotes. These are the documents:""" + self.concatenate_files(file_paths)

        return base_prompt

    def add_persona_modelling(self):
        self.prompting_strategies.append(
            '''You are an expert in the generation of structured JSON output from 
            unstructured data and in the analysis of English historical legal documents 
            in Middle Latin. Your target audience is historians well-versed in Medieval 
            English history.''')

    def add_context(self):
        self.prompting_strategies.append(
            '''You will be presented with documents from court rolls from the 13th and 14th 
            centuries which summarise court cases heard at courts in Medieval England. These 
            documents list all the persons involved in the court cases and contain information 
            about the locations they come from, their family relations, their professions and 
            their power relations with regard to other persons mentioned in the document. 
            The main language of the documents is medieval Latin but entities such as names, 
            locations and professions may be in Middle English.''')

    def add_iterative_approach(self):
        self.prompting_strategies.append(
            '''Take your time to read through the following instructions carefully. Take a deep 
            breath and take your time to work on the tasks step-by-step.''')

    def add_q_and_a_prompting(self):
        self.prompting_strategies.append(
            '''Ask questions if anything is unclear for a given step.'''
        )

    def add_schema_information(self):
        self.prompting_strategies.append('''
        This is the JSON schema:
        [
          {
            "id": ... ,
            "name": "...",
            "profession": "...",
            "family_relations": [
              {
                "relation_type": "...",
                "related_person": ...
              }
            ],
            "power_relations": [      {
                "relation_type": "...",
                "related_person": ...
              }],
            "place_of_origin": "...",
            "title": "",
            "org_role": ""
          }
        ]
        
        These are explanations of the categories in the JSON schema:
        * 'id': the unique id of the person
        * 'name': the name of the person
        * 'place of origin': the place the person comes from
        * 'profession': the work the person does
        * 'family relations' with 'relation type' (the type of family relation with another person 
         mentioned in the text) and 'related person' (the id of the person the named person is related
         to)
        * 'power relations' with 'relation type' (the type of power relation with another person 
         mentioned in the text) and 'related person' (the id of the person the named person is superior
         or subject to)
        * 'title': the title of the person
        * 'org_role': the role a person takes in an organisation''')

    def add_constraints(self):
        self.prompting_strategies.append('''While working on the tasks, pay attention to the 
        following rules:

        Rules:
        * Leave values empty if there is no information provided in the text.
        * Use the nominative singular form.
        * Stick to the spelling variations used in the document.
        * If word endings are cut due to OCR errors, reconstruct the complete word.
        * Use Latin terms for all of the values in the JSON file.
        * Put values preceded by the preposition 'de' into the place of origin category if they refer 
         to place names.
        * Check if the words after a name refer to a profession and if so, assign them to the 
        'profession' category.
        * Pay attention to the use of pronouns or attributes such as 'predictus' in the text to 
        make out persons already mentioned before.
        * List persons with the same names who are different persons.
        * Eliminate professions and places of origin from the 'name' category.''')

    def add_emotional_prompting(self):
        self.prompting_strategies.append('''It is very important for my research that you deliver 
        accurate results! I'll tip you $200 dollars for the best answer!''')

    def add_demonstrations(self, demonstrations):
        self.prompting_strategies.append('''Examples are contained in triple quotes. Consider these 
        examples for orientation:''' + self.concatenate_files(demonstrations))

