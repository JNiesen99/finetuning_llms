import re
from typing import Any, Optional

from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.evaluation import StringEvaluator
from langsmith import Client
from langchain.smith import RunEvalConfig, run_on_dataset
import pandas as pd
import os
from langchain.llms import HuggingFacePipeline
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    pipeline,
)
from peft import AutoPeftModelForCausalLM
import torch
from langchain import OpenAI
from google.cloud import storage
import shutil
from google.cloud import secretmanager
import os
from langchain_google_vertexai import VertexAI
from accelerate import Accelerator

class RelevanceEvaluator(StringEvaluator):
    """An LLM-based relevance evaluator."""

    def __init__(self):
        llm_gpt4 = ChatOpenAI(model="gpt-4", temperature=0)
        llm_gemini = VertexAI(model_name="gemini-pro")
        template = """ On a scale of 0 to 100, how well does the list of materials and tools in the OUTPUT match the project described in the INPUT?
        --------
        INPUT: {input}
        --------
        OUTPUT: {prediction}
        --------
        Reason step by step about why the score is appropriate, then print the score at the end. At the end, repeat that score alone on a new line."""

        self.eval_chain_gpt4 = LLMChain.from_string(llm=llm_gpt4, template=template)
        self.eval_chain_gemini = LLMChain.from_string(llm=llm_gemini, template=template)

    @property
    def requires_input(self) -> bool:
        return True

    @property
    def requires_reference(self) -> bool:
        return False

    @property
    def evaluation_name(self) -> str:
        return "scored_relevance"

    def _create_score_and_reasoning(self, text: str):
        reasoning, score = text.split("\n", maxsplit=1)
        score = re.search(r"\d+", score).group(0)
        if score is not None:
            score = float(score.strip()) / 100.0
        return {"score": score, "reasoning": reasoning.strip()}

    def _evaluate_strings(
        self,
        prediction: str,
        input: Optional[str] = None,
        reference: Optional[str] = None,
        **kwargs: Any
    ) -> dict:
        evaluator_result_gpt4 = self.eval_chain_gpt4(
            dict(input=input, prediction=prediction), **kwargs
        )
        evaluator_result_gemini = self.eval_chain_gemini(
            dict(input=input, prediction=prediction), **kwargs
        )
        gpt_4_text = self._create_score_and_reasoning(evaluator_result_gpt4["text"])
        gemini_text = self._create_score_and_reasoning(evaluator_result_gemini["text"])
        return {
            "score": (gpt_4_text["score"] + gemini_text["score"]) / 2,
            "reasoning": "gpt4 score: "
            + str(gpt_4_text["score"])
            + "gpt4: "
            + gpt_4_text["reasoning"]
            + "\ngemini score: "
            + str(gemini_text["score"])
            + "gemini: "
            + gemini_text["reasoning"],
        }


class Evaluator:
    """Evaluator class to evaluate LLMs using langchain/langsmith"""
    def __init__(
        self,
        model_path: str,
        model_name: str,
        test_dataset_path: str,
        project_name: str,
    ):
        if model_name == "Palm2":
            self.hf_pipeline_finetuned = VertexAI(model_name="text-bison@001")
        else:
            if model_name not in ["Llama2", "LeoLM"]:
                # download the finetuned model from cloud storage
                os.mkdir("model")
                storage_client = storage.Client()
                bucket_name = model_path.split("/")[2]
                folder_name = model_path.split(bucket_name + "/")[1]
                blobs = storage_client.list_blobs(bucket_name, prefix=folder_name)
                for blob in blobs:
                    filename = blob.name.split("/")[len(blob.name.split("/")) - 1]
                    blob.download_to_filename("model/" + filename)

                finetuned_model = AutoPeftModelForCausalLM.from_pretrained(
                    "model",
                    device_map="auto",  
                )

                finetuned_tokenizer = AutoTokenizer.from_pretrained("model")
                # delete downloaded model
                shutil.rmtree("model")
            else:
                finetuned_model = AutoModelForCausalLM.from_pretrained(
                    model_path, device_map="auto"
                )
                finetuned_tokenizer = AutoTokenizer.from_pretrained(model_path)

            # create hugging face pipeline
            pipeline_finetuned = pipeline(
                "text-generation",
                model=finetuned_model,
                tokenizer=finetuned_tokenizer,
                max_length=2000,
                temperature=0.5,
                config=finetuned_model.generation_config,
            )
            del finetuned_model
            del finetuned_tokenizer

            self.hf_pipeline_finetuned = HuggingFacePipeline(
                pipeline=pipeline_finetuned, model_id=model_name
            )

        # set langchain endpoint, api_key, tracing and project name
        os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
        os.environ["LANGCHAIN_API_KEY"] = "private"
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        self.project_name = project_name
        os.environ["LANGCHAIN_PROJECT"] = self.project_name

        # create langsmith client
        self.client = Client()

        # read in test dataset
        self.data_test = pd.read_json(
            test_dataset_path,
            lines=True,
            encoding="utf-8",
        )

        # get openai key from secret manager
        client = secretmanager.SecretManagerServiceClient()
        response = client.access_secret_version(
            request={
                "name": (
                    "private"
                    "private"
                    "versions/latest"
                )
            }
        )

        payload = response.payload.data.decode("UTF-8")

        os.environ["OPENAI_API_KEY"] = payload

    def _create_dataset(self, dataset_name: str, dataset_discription: str) -> None:
        """
        Function to create a langsmith dataset from runs
        """
        # check if the dataset already exists
        if self.client.has_dataset(dataset_name=dataset_name):
            self.client.delete_dataset(dataset_name=dataset_name)
        # Filter runs to add to the dataset
        dataset = self.client.create_dataset(
            dataset_name, description=dataset_discription
        )
        for index, row in self.data_test.iterrows():
            self.client.create_example(
                inputs={"input_text": row["input_text"]},
                outputs={"output_text": row["output_text"]},
                dataset_id=dataset.id,
            )

    def _evaluate_datasets(self, dataset_name: str, evaluation_config) -> None:
        """Function to call evaluators on a dataset"""
        # delete project if it exists
        try:
            self.client.delete_project(project_name=dataset_name + "evaluation")
        except:
            print("Project does not exist")
        self.client.run_on_dataset(
            dataset_name=dataset_name,
            llm_or_chain_factory=self.hf_pipeline_finetuned,
            evaluation=evaluation_config,
            project_name=dataset_name + "evaluation",
            verbose=True,
        )

    def evaluate(self, dataset_name: str) -> None:
        """
        Main evaluate function
        """
        # create evaluation config with the two methods
        evaluation_config = RunEvalConfig(
            evaluators=["embedding_distance"], custom_evaluators=[RelevanceEvaluator()]
        )

        # create dataset
        dataset_description = dataset_name
        self._create_dataset(dataset_name, dataset_description)

        # evaluate dataset
        self._evaluate_datasets(dataset_name, evaluation_config)
        del self.hf_pipeline_finetuned
