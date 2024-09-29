# historicalRelationExtraction
A repository for experiments regarding information extraction from historical texts.

## historicalPersonIdentification

### Repository Structure

The repository serves to conduct experiments regarding information extraction for person identification in historical texts.

It contains the data used for the experiments as well as functionality to conduct tool calling experiments for OpenAI models, fine-tune OpenAI and Mistral models and visualise results.

In order to use the functionality, the location of the root directory as well as the API keys for OpenAI and Mistral need to be set in the `.env` file.

### Data

The data set used in the experiments (i.e. plain text samples, ground truth data, demonstrations) can be found in the `data` folder.

### Tool Calling: JSON Generation

The entry point for conducting tool calling experiments is `experiment_function_calling_evaluation.py`. By triggering the `main` method, all the defined experiments can be started. The files used in the individual experiments are defined in the sample folders specified for the respective runs. The function specified for the experiment can be found in `ExtractJsonFromPlainText.py`. Functionality to evaluate the JSON results is provided in `evaluation`. The results of the tool calling experiments can be found in `evaluation_results`.

### Tool Calling: Finding Best Person Matches

The function for finding the n persons most similar to a target person specified in the input an triggered by running the in `experiment_function_calling_finding_person_in_jsons.py`. The folder with the predicted files analysed can be set in the `run` method of `FindPersonInJsonFiles.py`.

### Fine-Tuning

The fine-tuning functionality is located in `fine_tuning`. The fine-tuning results for OpenAI and Mistral (i.e. the metrics documenting the training process) are found in `fine_tuning_files`.

### Visualisation

Functionality for visualising the results as well as the generated plots are located in the `visualisation` folder.
