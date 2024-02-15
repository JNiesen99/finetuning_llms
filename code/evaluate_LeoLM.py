from evaluator import Evaluator

evaluator = Evaluator(
    model_path="LeoLM/leo-hessianai-13b-chat",
    model_name="LeoLM",
    test_dataset_path="finetune-llms/data/projectquestion_LeoLM_testing.jsonl",
    project_name="projectquestions",
)

# start the model evaluation
evaluator.evaluate(dataset_name="LeoLM")
