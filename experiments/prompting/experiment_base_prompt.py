from experiments.prompting.super_prompting_experiment import BaseExperimentPrompting


class ExperimentBasePrompt(BaseExperimentPrompting):

    def experiment_base_prompt(self):
        prompt = self.prompt_builder.add_base_prompt([self.test_files]).build_prompt()
        return prompt, "base_prompt"

    def run(self):
        prompt, filename = self.experiment_base_prompt()
        self.dialogue.prompt_assistant_response(prompt, filename)

        while True:
            self.dialogue.add_dynamic_prompting(filename)


if __name__ == '__main__':
    experiment_base_prompt = ExperimentBasePrompt()
    experiment_base_prompt.run()
