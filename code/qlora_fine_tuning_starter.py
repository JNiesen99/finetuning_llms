from finetuner import Finetuner
from evaluator import Evaluator

# load the base model and the fine-tuning dataset
finetuner = Finetuner(
    preprocessing="standard",
    base_model="LeoLM/leo-hessianai-13b-chat",
    quantization=True,
    dataset_path="finetune-llms/data/projectquestion_LeoLM_training.jsonl",
    dataset_columns=["input_text", "output_text"],
)

# start the finetuning
finetuner.fine_tuning(
    method="lora",
    fine_tuning_parameters="finetune-llms/hyperparameters/lora_parameters.json",
    training_parameters="finetune-llms/hyperparameters/training_parameters.json",
    adapter_output_dir="finetune-llms/models/adapters_LeoLM_qlora",
)

# create a new evaluator
evaluator = Evaluator(
    model_path="gs://finetuning_data_eu_west4/masterarbeit/models/adapters_LeoLM_qlora",
    model_name="LeoLM_qlora",
    test_dataset_path="finetune-llms/data/projectquestion_LeoLM_testing.jsonl",
    project_name="projectquestions",
)

# start the model evaluation
evaluator.evaluate(dataset_name="LeoLM_qlora")


# Llama2 ------------------------------------------------------------------------------------
finetuner = Finetuner(
    preprocessing="standard",
    base_model="meta-llama/Llama-2-13b-chat-hf",
    quantization=True,
    dataset_path="finetune-llms/data/projectquestion_Llama2_training.jsonl",
    dataset_columns=["input_text", "output_text"],
)

# start the finetuning
finetuner.fine_tuning(
    method="lora",
    fine_tuning_parameters="finetune-llms/hyperparameters/lora_parameters.json",
    training_parameters="finetune-llms/hyperparameters/training_parameters.json",
    adapter_output_dir="finetune-llms/models/adapters_Llama2_qlora",
)

# create a new evaluator
evaluator = Evaluator(
    model_path="gs://finetuning_data_eu_west4/masterarbeit/models/adapters_Llama2_qlora",
    model_name="Llama2_qlora",
    test_dataset_path="finetune-llms/data/projectquestion_Llama2_testing.jsonl",
    project_name="projectquestions",
)

# start the model evaluation
evaluator.evaluate(dataset_name="Llama2_qlora")
