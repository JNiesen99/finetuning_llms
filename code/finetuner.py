from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
from peft import (
    LoraConfig,
    get_peft_model,
    AutoPeftModelForCausalLM,
    PrefixTuningConfig,
    TaskType,
    prepare_model_for_kbit_training,
    AdaptionPromptConfig,
    AdaptionPromptModel,
)
from datasets import load_from_disk
from datasets import Dataset
import transformers
from transformers import DefaultFlowCallback
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
import json
from google.cloud import storage
import os
import glob
import shutil
from accelerate import Accelerator, infer_auto_device_map, dispatch_model
from accelerate.utils import get_balanced_memory
from prefix_tuning import prefix_tuning, prefix_preprocessing

class Finetuner:
    """Finetuner class to finetune llms with various methods"""
    def __init__(
        self,
        base_model: str,
        quantization: bool,
        dataset_path: str,
        preprocessing: str,
        dataset_columns: list = ["input", "output"],
    ) -> None:
        # create tensorboard writer
        self.writer = SummaryWriter("runs")
        # clear gpu cache
        torch.cuda.empty_cache()

        # for multi gpu processing
        current_device = Accelerator().process_index
        self.accelerator = Accelerator()

        if quantization:
            # read the quantized model
            # define bits and bytes config
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )

            # load base LLM model
            self.model = AutoModelForCausalLM.from_pretrained(
                base_model,
                quantization_config=bnb_config,
                device_map={"": current_device},
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                base_model, device_map="auto"
            )
            max_memory = get_balanced_memory(
                self.model,
                max_memory=None,
                no_split_module_classes=[
                    "LlamaDecoderLayer",
                    "LlamaSdpaAttention",
                    "LlamaRotaryEmbedding",
                    "LlamaMLP",
                    "Linear",
                    "SiLU",
                    "LlamaRMSNorm",
                ],
                dtype="float16",
                low_zero=False,
            )

            device_map = infer_auto_device_map(
                model=self.model,
                max_memory=max_memory,
                no_split_module_classes=[
                    "LlamaDecoderLayer",
                    "LlamaSdpaAttention",
                    "LlamaRotaryEmbedding",
                    "LlamaMLP",
                    "Linear",
                    "SiLU",
                    "LlamaRMSNorm",
                ],
                dtype="float16",
            )
            self.model = dispatch_model(self.model, device_map=device_map)

        # load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)

        # load dataset
        self.data = pd.read_json(
            dataset_path,
            lines=True,
            encoding="utf-8",
        )
        if preprocessing == "input/output":
            self.dataset = prefix_preprocessing(
                self.data, self.tokenizer, dataset_columns
            )
        else:
            # create list of input output strings
            dataset_text = self.data[dataset_columns].astype(str).sum(axis=1)

            # create huggingface Dataset
            self.dataset = Dataset.from_dict({"text": dataset_text})

            self.tokenizer.pad_token = self.tokenizer.eos_token

            # tokenize text
            self.dataset = self.dataset.map(
                lambda sampeles: self.tokenizer(sampeles["text"]),
                batched=True,
            )

    def _upload_to_cs(self, model_path, bucket_name, dest_path) -> None:
        """
        Function to upload model / model adapters to cloud storage
        """
        storage_client = storage.Client()
        bucket = storage_client.get_bucket(bucket_name)
        for item in glob.glob(model_path + "/*"):
            if os.path.isfile(item):
                blob = bucket.blob(
                    os.path.join(
                        dest_path
                        + "/"
                        + model_path.split("/")[len(model_path.split("/")) - 1],
                        os.path.basename(item),
                    )
                )
                blob.upload_from_filename(item)

    def _print_trainable_parameters(self) -> None:
        """
        Prints the number of trainable parameters in the model.
        """
        trainable_params = 0
        all_param = 0
        for _, param in self.model.named_parameters():
            # add number of parameters for that layer
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        print(
            f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
        )

    def _lora_finetuning(
        self,
        fine_tuning_parameters_path: str,
    ) -> None:
        """
        Function to prepare a model for lora finetuning
        """
        # prepare model for kbit training
        self.model.gradient_checkpointing_enable()
        self.model = prepare_model_for_kbit_training(self.model)

        # get the fine-tuning parameters
        with open(fine_tuning_parameters_path) as json_file:
            parameters = json.load(json_file)

        # define Lora Config
        config = LoraConfig(
            r=parameters["rank"],
            # target_modules = ["q_proj", "v_proj"], # only targeting attention blocks of the model
            target_modules=parameters["target_modules"],
            lora_alpha=parameters["lora_alpha"],
            lora_dropout=parameters["lora_dropout"],
            bias=parameters["bias"],
            task_type=parameters["task_type"],
        )
        # transform model to peft model with lora config
        self.model = get_peft_model(self.model, config)

        # print trainable parameters
        self._print_trainable_parameters()

    def _prefix_tuning(
        self,
        fine_tuning_parameters_path: str,
        training_parameters: str,
    ) -> None:
        """
        Function to prepare a model for prefix finetuning
        """
        # get the fine-tuning parameters
        with open(fine_tuning_parameters_path) as json_file:
            ft_parameters = json.load(json_file)

        # get the training parameters
        with open(training_parameters) as json_file:
            tr_parameters = json.load(json_file)

        self.model, self.tokenizer = prefix_tuning(
            self.dataset, self.tokenizer, self.model, ft_parameters, tr_parameters
        )

    def _adapter_tuning(self, fine_tuning_parameters_path) -> None:
        """
        Function to prepare a model for adapter finetuning
        """
        # get the fine-tuning parameters
        with open(fine_tuning_parameters_path) as json_file:
            parameters = json.load(json_file)

        # define Lora Config
        adapter_config = AdaptionPromptConfig(
            task_type=parameters["task_type"],
            adapter_layers=parameters["adapter_layers"],
            adapter_len=parameters["adapter_len"],
        )
        # transform model to peft model with lora config
        self.model = get_peft_model(self.model, adapter_config)

        # print trainable parameters
        self._print_trainable_parameters()

    def fine_tuning(
        self,
        method: str,
        fine_tuning_parameters: str,
        training_parameters: dict,
        adapter_output_dir: str,
        # model_output_dir: str,
        bucket_name: str = "finetuning_data_eu_west4",
        dest_path: str = "masterarbeit/models",
    ) -> None:
        """
        Calls the different finetuning functions
        """
        if method == "prefix":
            self._prefix_tuning(fine_tuning_parameters, training_parameters)
        else:
            if method == "lora":
                self._lora_finetuning(fine_tuning_parameters)
            elif method == "adapter":
                self._adapter_tuning(fine_tuning_parameters)

            # get the training parameters
            with open(training_parameters) as json_file:
                parameters = json.load(json_file)

            # train the model
            trainer = transformers.Trainer(
                model=self.model,
                train_dataset=self.dataset,
                args=transformers.TrainingArguments(
                    per_device_train_batch_size=parameters[
                        "per_device_train_batch_size"
                    ],
                    gradient_accumulation_steps=parameters[
                        "gradient_accumulation_steps"
                    ],
                    warmup_steps=parameters["warmup_steps"],
                    logging_steps=parameters["logging_steps"],
                    max_steps=parameters["max_steps"],
                    learning_rate=parameters["learning_rate"],
                    fp16=parameters["fp16"],
                    output_dir=adapter_output_dir,
                    optim=parameters["optim"],
                    report_to=parameters["report_to"],
                ),
                callbacks=[DefaultFlowCallback()],
                data_collator=transformers.DataCollatorForLanguageModeling(
                    self.tokenizer, mlm=False
                ),
            )
            self.model.config.use_cache = (
                False  # silence the warnings. Re-enable for inference
            )
            trainer.train()

            # save the model
        self.model.save_pretrained(adapter_output_dir)

        # save tokenizer
        self.tokenizer.save_pretrained(adapter_output_dir)

        # upload model to cs
        self._upload_to_cs(
            model_path=adapter_output_dir,
            bucket_name=bucket_name,
            dest_path=dest_path,
        )
        # delete adapter folder
        shutil.rmtree(adapter_output_dir)
        del self.model
        del self.tokenizer
