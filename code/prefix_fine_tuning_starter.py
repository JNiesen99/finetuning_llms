from finetuner import Finetuner
from evaluator import Evaluator

# load the base model and the fine-tuning dataset
finetuner = Finetuner(
    preprocessing="input/output",
    base_model="LeoLM/leo-hessianai-13b-chat",
    quantization=False,
    dataset_path="finetune-llms/data/projectquestion_LeoLM_training.jsonl",
    dataset_columns=["input_text", "output_text"],
)

# start the finetuning
finetuner.fine_tuning(
    method="prefix",
    fine_tuning_parameters="finetune-llms/hyperparameters/prefix_parameters.json",
    training_parameters="finetune-llms/hyperparameters/training_parameters.json",
    adapter_output_dir="finetune-llms/models/adapters_LeoLM_prefix",
)

# create a new evaluator
evaluator = Evaluator(

    model_path="gs://finetuning_data_eu_west4/masterarbeit/models/adapters_LeoLM_prefix",
    model_name="LeoLM_prefix",
    test_dataset_path="finetune-llms/data/projectquestion_LeoLM_testing.jsonl",
    project_name="projectquestions",
)

# start the model evaluation
evaluator.evaluate(dataset_name="LeoLM_prefix")


# Llama2 -----------------------------------------------------------------------------------------
finetuner = Finetuner(
    preprocessing="input/output",
    base_model="meta-llama/Llama-2-13b-chat-hf",
    quantization=True,
    dataset_path="finetune-llms/data/projectquestion_Llama2_training.jsonl",
    dataset_columns=["input_text", "output_text"],
)

# start the finetuning
finetuner.fine_tuning(
    method="prefix",
    fine_tuning_parameters="finetune-llms/hyperparameters/prefix_parameters.json",
    training_parameters="finetune-llms/hyperparameters/training_parameters.json",
    adapter_output_dir="finetune-llms/models/adapters_llama2_prefix",
)

# create a new evaluator
evaluator = Evaluator(
    model_path="gs://finetuning_data_eu_west4/masterarbeit/models/adapters_llama2_prefix",
    model_name="Llama2_prefix",
    test_dataset_path="finetune-llms/data/projectquestion_Llama2_testing.jsonl",
    project_name="projectquestions",
)

# start the model evaluation
evaluator.evaluate(dataset_name="Llama2_prefix")
