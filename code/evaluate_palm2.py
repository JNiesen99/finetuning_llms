from evaluator import Evaluator

evaluator = Evaluator(
    model_path=" ",
    model_name="Palm2",
    test_dataset_path="finetune-llms/data/projectquestion_Llama2_testing.jsonl",
    project_name="projectquestions",
)

# start the model evaluation
evaluator.evaluate(dataset_name="Palm2")
