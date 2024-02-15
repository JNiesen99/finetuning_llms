import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from peft import (
    get_peft_model,
    PrefixTuningConfig,
)
from datasets import Dataset, DatasetDict
from transformers import (
    get_linear_schedule_with_warmup,
    default_data_collator,
)
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


# data preprocessing
def prefix_preprocessing(data, tokenizer, dataset_columns):
    """
    Creates the training dataset and starts the prprocessing
    """
    text_column = dataset_columns[0]
    label_column = dataset_columns[1]
    # create dataset
    dataset = DatasetDict({"train": Dataset.from_pandas(data)})

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    def preprocess_function(examples):
        """
        Preprocesses the finetuning data, the input and output must be concatenated.
        """
        max_length = 568
        batch_size = len(examples[text_column])
        inputs = [f"{text_column} : {x} Label" for x in examples[text_column]]
        targets = [str(x) for x in examples[label_column]]
        model_inputs = tokenizer(inputs)
        labels = tokenizer(
            targets, add_special_tokens=False
        )  # don't add bos token because we concatenate with inputs
        for i in range(batch_size):
            sample_input_ids = model_inputs["input_ids"][i]
            label_input_ids = labels["input_ids"][i] + [tokenizer.eos_token_id]
            # print(i, sample_input_ids, label_input_ids)
            model_inputs["input_ids"][i] = sample_input_ids + label_input_ids
            labels["input_ids"][i] = [-100] * len(sample_input_ids) + label_input_ids
            model_inputs["attention_mask"][i] = [1] * len(model_inputs["input_ids"][i])
        # print(model_inputs)
        for i in range(batch_size):
            sample_input_ids = model_inputs["input_ids"][i]
            label_input_ids = labels["input_ids"][i]
            model_inputs["input_ids"][i] = [tokenizer.pad_token_id] * (
                max_length - len(sample_input_ids)
            ) + sample_input_ids
            model_inputs["attention_mask"][i] = [0] * (
                max_length - len(sample_input_ids)
            ) + model_inputs["attention_mask"][i]
            labels["input_ids"][i] = [-100] * (
                max_length - len(sample_input_ids)
            ) + label_input_ids
            model_inputs["input_ids"][i] = torch.tensor(
                model_inputs["input_ids"][i][:max_length]
            )
            model_inputs["attention_mask"][i] = torch.tensor(
                model_inputs["attention_mask"][i][:max_length]
            )
            labels["input_ids"][i] = torch.tensor(labels["input_ids"][i][:max_length])
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    # map the preprocessing function (tokenization) batch-wise on the dataset
    processed_datasets = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset["train"].column_names,
        load_from_cache_file=False,
        desc="Running tokenizer on dataset",
    )

    return processed_datasets


def prefix_tuning(dataset, tokenizer, model, ft_parameters, tr_parameters):
    """
        Fucntion that starts the prefix tuning process via pytorch
    """
    batch_size = tr_parameters["per_device_train_batch_size"]
    lr = tr_parameters["learning_rate"]
    num_epochs = tr_parameters["max_steps"]

    train_dataset = dataset["train"]
    eval_dataset = dataset["train"]

    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=default_data_collator,
        batch_size=batch_size,
        pin_memory=True,
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        collate_fn=default_data_collator,
        batch_size=batch_size,
        pin_memory=True,
    )

    peft_config = PrefixTuningConfig(
        peft_type=ft_parameters["peft_type"],
        task_type=ft_parameters["task_type"],
        num_virtual_tokens=ft_parameters["num_virtual_tokens"],
    )

    model = get_peft_model(model, peft_config)

    def print_trainable_parameters(model) -> None:
        """
        Prints the number of trainable parameters in the model.
        """
        trainable_params = 0
        all_param = 0
        for _, param in model.named_parameters():
            # add number of parameters for that layer
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        print(
            f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
        )

    print_trainable_parameters(model)

    # model
    # optimizer and lr scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=(len(train_dataloader) * num_epochs),
    )

    # create summary write
    writer = SummaryWriter()
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for step, batch in enumerate(tqdm(train_dataloader)):
            outputs = model(**batch)
            loss = outputs.loss
            total_loss += loss.detach().float()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        model.eval()
        eval_loss = 0
        eval_preds = []
        for step, batch in enumerate(tqdm(eval_dataloader)):
            with torch.no_grad():
                outputs = model(**batch)
            loss = outputs.loss
            eval_loss += loss.detach().float()
            eval_preds.extend(
                tokenizer.batch_decode(
                    torch.argmax(outputs.logits, -1).detach().cpu().numpy(),
                    skip_special_tokens=True,
                )
            )

        eval_epoch_loss = eval_loss / len(eval_dataloader)
        eval_ppl = torch.exp(eval_epoch_loss)
        train_epoch_loss = total_loss / len(train_dataloader)
        train_ppl = torch.exp(train_epoch_loss)
        print(
            f"{epoch=}: {train_ppl=} {train_epoch_loss=} {eval_ppl=} {eval_epoch_loss=}"
        )
        writer.add_scalar("train/loss", train_epoch_loss, epoch + 1)
        writer.add_scalar("test/loss", eval_epoch_loss, epoch + 1)
    return model, tokenizer
