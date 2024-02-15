from evaluator import Evaluator

evaluator = Evaluator(
    model_path="meta-llama/Llama-2-13b-chat-hf",
    model_name="Llama2",
    test_dataset_path="finetune-llms/data/projectquestion_Llama2_testing.jsonl",
    project_name="projectquestions",
)

# start the model evaluation
evaluator.evaluate(dataset_name="Llama2")
