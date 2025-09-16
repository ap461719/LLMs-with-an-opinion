"""
QLoRA fine-tuning pipeline for LLaMA models using PEFT and Hugging Face Transformers.

This module provides a callable `finetune()` function that prepares a LLaMA model
with 4-bit quantization and LoRA adapters, and trains it on a provided dataset of prompts.


"""

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import load_dataset, concatenate_datasets, DatasetDict
import random

def tokenize_function_llama(example):
    """
    Tokenizes dataset examples for LLaMA fine-tuning.

    This function concatenates the prompt and pads/truncates to max length 256.
    The labels are set to be the same as input_ids for causal language modeling.

    Args:
        example (dict): A dataset example containing 'prompt' key.

    Returns:
        dict: Tokenized input_ids and labels.
    """
    if auto_tokenizer.pad_token is None:
        auto_tokenizer.pad_token = auto_tokenizer.eos_token

    outputs = auto_tokenizer(
        example["prompt"],
        truncation=True,
        padding="max_length",
        max_length=256,
    )
    outputs["labels"] = outputs["input_ids"]  # align labels exactly
    return outputs

def finetune(
    all_prompts_dataset,
    model_name="meta-llama/Llama-2-7b-hf",
    output_dir="llama-7b-qlora-finetuned",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=3
):
    """
    Fine-tunes a LLaMA model using QLoRA + PEFT on a dataset of text prompts.

    This function loads a pretrained LLaMA model in 4-bit precision,
    attaches LoRA adapters, tokenizes the dataset, and trains the model.

    Args:
        all_prompts_dataset (datasets.DatasetDict): Dataset containing 'train' and 'test' splits with 'prompt' column.
        model_name (str): Pretrained model name or local path.
        output_dir (str): Directory to save the fine-tuned model.
        per_device_train_batch_size (int): Batch size per device for training.
        per_device_eval_batch_size (int): Batch size per device for evaluation.
        gradient_accumulation_steps (int): Number of steps to accumulate gradients before updating weights.
        learning_rate (float): Learning rate for optimizer.
        num_train_epochs (int): Number of epochs to train.

    Returns:
        None
    """
    global auto_tokenizer

    # Load Llama tokenizer from Hugging Face
    auto_tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    auto_tokenizer.pad_token = auto_tokenizer.eos_token

    # BitsAndBytesConfig (4-bit QLoRA)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype="bfloat16"
    )

    # Load model in 4-bit + prepare for QLoRA
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    model = prepare_model_for_kbit_training(model)

    # LoRA config
    peft_config = LoraConfig(
        r=8,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, peft_config)

    # Re-tokenize cleanly
    aug_tokenized_dataset_llama = all_prompts_dataset.map(
        tokenize_function_llama, 
        batched=True, 
        remove_columns=["prompt"]
    )
    aug_tokenized_dataset_llama.set_format("torch")

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        fp16=True,
        save_total_limit=2,
        push_to_hub=False,
        report_to="tensorboard",
        label_names=["label"]
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=auto_tokenizer,
        mlm=False
    )

    llama_trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=aug_tokenized_dataset_llama["train"],
        eval_dataset=aug_tokenized_dataset_llama["test"],
        tokenizer=auto_tokenizer,
        data_collator=data_collator
    )

    llama_trainer.train()
    

def add_prompt(example, positive_txt="positive: ", negative_text="negative: "):
    """
    Adds a basic positive or negative prefix to the example text.
    """
    if example["label"] == 1:
        prompt = positive_txt
    elif example["label"] == 0:
        prompt = negative_text
    else:
        prompt = ""
    return {"prompt": prompt + " " + example["text"]}

def add_augmented_prompt(example, positive_prompts, negative_prompts):
    """
    Adds a randomly chosen detailed prompt to the example text.
    """
    if example["label"] == 1:
        prompt = random.choice(positive_prompts)
    elif example["label"] == 0:
        prompt = random.choice(negative_prompts)
    else:
        prompt = ""
    return {"prompt": prompt + " " + example["text"]}

def create_augmented_imdb_dataset(output_dir="data", seed=42, num_augmented=10000):
    """
    Loads the IMDB dataset, creates base and augmented prompt datasets,
    and saves CSVs of raw data. Returns a DatasetDict of all prompts.

    Args:
        output_dir (str): Path to save raw CSV files.
        seed (int): Random seed for reproducibility.
        num_augmented (int): Number of examples to select for augmentation.

    Returns:
        DatasetDict: A dataset dict with 'train', 'test', 'unsupervised' splits.
    """
    dataset = load_dataset("imdb")

    # Split dataset
    train_data = dataset["train"]
    test_data = dataset["test"]

    # Save raw train/test CSVs
    train_df = train_data.to_pandas()
    test_df = test_data.to_pandas()
    train_df.to_csv(f"{output_dir}/imdb_train.csv", index=False)
    test_df.to_csv(f"{output_dir}/imdb_test.csv", index=False)

    # Print stats
    train_pos_df = train_df[train_df["label"] == 1]
    train_neg_df = train_df[train_df["label"] == 0]
    print(f"Number of positive reviews in training set: {train_pos_df.shape[0]}")
    print(f"Number of negative reviews in training set: {train_neg_df.shape[0]}")

    # Create base prompts dataset
    base_prompts_dataset = dataset.map(add_prompt)

    train_prompts = base_prompts_dataset["train"].remove_columns(["text", "label"])
    test_prompts = base_prompts_dataset["test"].remove_columns(["text", "label"])
    train_prompts.to_csv(f"{output_dir}/imdb_train_with_prompt.csv", index=False)
    test_prompts.to_csv(f"{output_dir}/imdb_test_with_prompt.csv", index=False)

    # Define augmentation prompts
    positive_prompts = [
        "Write a positive movie review:",
        "Praise this movie:",
        "Explain why the movie was excellent:",
        "Give a glowing review of the film:",
        "What made this movie so enjoyable?",
        "Highlight the strengths of the film:",
        "Be optimistic about the movie:",
        "Why is this a must-watch?",
        "Describe the movie in a cheerful way:",
        "What did you love about the movie?",
        "This movie deserves a standing ovation because",
    ]

    negative_prompts = [
        "Write a negative movie review:",
        "Criticize this movie:",
        "Explain why the movie was terrible:",
        "Be harsh about the film:",
        "What made this movie so disappointing?",
        "Highlight the flaws of the film:",
        "Be pessimistic about the movie:",
        "Why is this a movie to skip?",
        "Describe the movie in a critical way:",
        "What did you dislike about the movie?",
        "This movie deserves a thumbs down because",
    ]

    # Create augmented datasets
    augmented_train_set = dataset["train"].shuffle(seed=seed).map(
        lambda ex: add_augmented_prompt(ex, positive_prompts, negative_prompts)
    ).select(range(num_augmented))

    augmented_test_set = dataset["test"].shuffle(seed=seed).map(
        lambda ex: add_augmented_prompt(ex, positive_prompts, negative_prompts)
    ).select(range(num_augmented))

    augmented_us_set = dataset["unsupervised"].shuffle(seed=seed).map(
        lambda ex: add_augmented_prompt(ex, positive_prompts, negative_prompts)
    ).select(range(num_augmented))

    augmented_train_dataset = augmented_train_set.remove_columns(["text", "label"])
    augmented_test_dataset = augmented_test_set.remove_columns(["text", "label"])
    augmented_us_dataset = augmented_us_set.remove_columns(["text", "label"])

    # Combine base + augmented into final DatasetDict
    all_prompts_dataset = DatasetDict({
        "train": concatenate_datasets([train_prompts, augmented_train_dataset]),
        "test": concatenate_datasets([test_prompts, augmented_test_dataset]),
        "unsupervised": concatenate_datasets([
            base_prompts_dataset["unsupervised"].remove_columns(["text", "label"]),
            augmented_us_dataset
        ]),
    })

    return all_prompts_dataset
