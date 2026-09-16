from pathlib import Path


class PromptBuilder:
    def __init__(self):
        self.prompting_strategies: list[str] = []

    @staticmethod
    def concatenate_files(file_paths, section_name):
        sections = []

        for file_path in file_paths:
            path = Path(file_path)

            try:
                content = path.read_text(encoding='utf-8').strip()
            except OSError as error:
                raise OSError(f"Could not read file: {path}") from error

            sections.append(
                f"<{section_name}>\n"
                f"{content}\n"
                f"</{section_name}>"
            )

        return "\n\n".join(sections)

    def add_base_prompt(self, file_paths):
        prompt = (
                "Extract the persons and their relationships from the court document "
                "and transfer the result to the required structured output by calling the provided tool.\n"
                "The court document is enclosed in <court_document> tags:\n"
                + self.concatenate_files(file_paths, 'court_document')
        )
        self.prompting_strategies.append(prompt)
        return self

    def add_task(self):
        prompt = (
            "Extract the persons and their relationships from the court document "
            "and transfer the result to the required structured output by calling the provided tool.\n"
        )
        self.prompting_strategies.append(prompt)
        return self

    def add_task_without_tool_calling(self):
        prompt = (
            "Extract the persons and their relationships from the court document "
            "and transfer the result to a JSON object matching the provided output structure. "
            "Return only the JSON object, without Markdown or additional text.\n"
        )
        self.prompting_strategies.append(prompt)
        return self

    def add_input_text(self, file_paths):
        prompt = (
                "Process the following court document, enclosed in <court_document> tags:\n" +
                self.concatenate_files(file_paths, 'court_document')
        )
        self.prompting_strategies.append(prompt)
        return self

    def add_persona_modelling(self):
        prompt = (
            "You are an expert in structured information extraction from medieval English legal documents written primarily in medieval Latin.\n"
        )
        self.prompting_strategies.append(prompt)
        return self

    def add_context(self):
        prompt = (
            "You will be presented with 13th- and 14th-century English court-roll documents. "
            "They mention persons and information such as names, places of origin, professions, titles, family relations and legal relationships or roles. "
            "The documents are primarily written in medieval Latin, but names, places and professions may also occur in Middle English.\n"
        )
        self.prompting_strategies.append(prompt)
        return self

    def add_iterative_approach(self):
        prompt = (
            "Read the instructions carefully and work through the extraction step by step.\n"
        )
        self.prompting_strategies.append(prompt)
        return self

    def add_schema_information(self):
        prompt = (
            "Use the following output structure:\n"
            "{\n"
            "  \"person_list\": [\n"
            "    {\n"
            "      \"id\": ...,\n"
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
            "      \"title\": \"...\"\n"
            "    }\n"
            "  ]\n"
            "}\n\n"
            "Field definitions:\n"
            "* 'id': unique integer identifier for the person within the document.\n"
            "* 'name': first name of the person.\n"
            "* 'cognomen': additional identifying name or byname.\n"
            "* 'profession': profession or occupation.\n"
            "* 'family_relations': family relationships of the person.\n"
            "* 'legal_relationship': legal or procedural relationships and roles of the person.\n"
            "* 'relation_type': relationship or role of the person whose record is being described.\n"
            "* 'related_person': id of the related person, or null when the relation is explicit but the other person cannot be linked to an extracted person.\n"
            "* 'place_of_origin': place the person comes from.\n"
            "* 'title': formal secular or ecclesiastical title.\n"
        )
        self.prompting_strategies.append(prompt)
        return self

    def add_constraints(self):
        prompt = (
            "While performing the extraction, follow these rules:\n"
            "* Extract only information supported by the source text. Do not infer missing information.\n"
            "* Assign every person a unique integer 'id' within the document.\n"
            "* Use \"\" for missing string values and [] when no family or legal relations are given.\n"
            "* Use null for 'related_person' only when a relation is explicit but the other person cannot be linked to an extracted person.\n"
            "* Convert clearly inflected Latin personal names to the nominative singular when the nominative form is unambiguous, for example 'Alexandrum' to 'Alexandrus' and 'Willelmum' to 'Willelmus'. If the nominative form is uncertain, preserve the source form.\n"
            "* Normalize capitalization in extracted values by capitalizing the first character of the value. "
            "When a cognomen begins with 'le' or 'de', write it as 'Le' or 'De'.\n"
            "* Otherwise preserve the spelling used in the source apart from required nominative conversion and unambiguous OCR correction.\n"
            "* Correct obvious OCR errors, including truncated, split or misrecognized characters, only when the intended form is unambiguous; otherwise preserve the source form.\n"
            "* Do not translate names, cognomina, places, professions or titles. Use the specified Latin labels for 'relation_type'.\n"
            "* Use only the first name in 'name'.\n"
            "* Use 'cognomen' for an additional identifying name or byname following the first name. "
            "Keep 'Le' or 'De' when they are part of the cognomen. "
            "A cognomen may also provide information represented separately as a profession or place of origin.\n"
            "* Use 'profession' for occupations. If an occupation is introduced by 'le', omit 'le' from the profession value.\n"
            "* Use 'place_of_origin' when 'de' introduces a place of origin, and omit 'de' from the value.\n"
            "* Use 'title' for formal secular or ecclesiastical titles or offices. Do not also place a title in another category unless the text independently supports that category.\n"
            "* Do not include an associated place in 'title' when the place is given separately as information about the person; store it in 'place_of_origin' instead. "
            "Keep geographical or institutional wording in 'title' when it is an integral part of the title itself.\n"
            "* Pay attention to pronouns and references such as 'predictus' when deciding whether a person has already been mentioned.\n"
            "* Do not merge different persons solely because they have the same name.\n"
            "* Every 'relation_type' describes the role of the person whose record contains the relation.\n"
            "* Represent reciprocal relationships from both perspectives only when both roles are supported by the text.\n"
            "* Do not add a reciprocal legal role solely because a corresponding paired role exists.\n"
            "* One-sided roles are assigned only to the person who holds that role.\n"
            "* Allowed family relation types are: Pater, Mater, Frater, Soror, Filius, Filia, Avus, Avia, Avunculus, Consanguineus, Noverca, Vitricus, Privignus, Privigna, Matertera, Patruus, Amita, Nepos, Neptis, Maritus, Uxor, Progenitor, Progenies.\n"
            "* Allowed legal relation types are: Tenens, Dominus Feodi, Testator, Heres, Plegiarius, Attornatus, Principalis, Essoniator, Particeps, Reus, Petitor, Custos, Pupillus, Warantus.\n"
        )

        self.prompting_strategies.append(prompt)
        return self

    def add_emotional_prompting(self):
        prompt = (
            "Accurate extraction is very important for this research, so check your results carefully.\n"
        )
        self.prompting_strategies.append(prompt)
        return self

    def add_demonstrations(self, demonstrations):
        prompt = (
                "Use the following example only as guidance for interpretation and output structure. "
                "Each example is enclosed in <example> tags:\n" +
                self.concatenate_files(demonstrations, 'example')
        )
        self.prompting_strategies.append(prompt)
        return self

    def build_prompt(self):
        return '\n\n'.join(strategy.strip() for strategy in self.prompting_strategies)
