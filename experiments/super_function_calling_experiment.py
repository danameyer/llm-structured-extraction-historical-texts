from function_calling_setup.dialogue_completion import DialogueCompletion
from function_calling_setup.providers.base_provider import LLMProvider


class BaseExperimentFunctionCalling:
    def __init__(
            self,
            test_files,
            demonstrations,
            experiment_dir,
            pred_file_name,
            provider: LLMProvider,
            dialogue_cls=DialogueCompletion,
    ):
        self.experiment_dir = experiment_dir
        self.dialogue = dialogue_cls(provider=provider, experiment_dir=experiment_dir)
        self.test_files = test_files
        self.demonstrations = demonstrations
        self.pred_file_name = pred_file_name
        self.model_name = provider.model

    def run(self):
        pass
