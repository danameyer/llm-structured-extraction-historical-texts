from experiments.prompting.super_prompting_experiment import BaseExperimentPrompting


class ExperimentAllPrinciplesFewShot(BaseExperimentPrompting):

    def experiment_all_principles_few_shot(self):
        prompt = (self.prompt_builder.add_persona_modelling()
                  .add_context()
                  .add_iterative_approach()
                  .add_q_and_a_prompting()
                  .add_base_prompt([self.test_files])
                  .add_demonstrations([self.demonstrations])
                  .add_schema_information()
                  .add_constraints()
                  .add_emotional_prompting()
                  .build_prompt())
        return prompt, "all_principles_few_shot"

    def run(self):
        prompt, filename = self.experiment_all_principles_few_shot()
        self.dialogue.prompt_assistant_response(prompt, filename)

        while True:
            self.dialogue.add_dynamic_prompting(filename)


if __name__ == '__main__':
    experiment_all_principlesFewShot = ExperimentAllPrinciplesFewShot("/tmp/experiments/experiment_all_principles_few_shot")
    experiment_all_principlesFewShot.run()
