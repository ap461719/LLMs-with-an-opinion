"""
Instruction-Tuning Script for GPT-2 on IMDB Movie Reviews

This module provides a finetune() function that instruction-tunes a Hugging Face GPT-2 model
on a custom dataset of movie summaries and reviews using prompt-based instruction templates.
It uses Hugging Face datasets, transformers Trainer API, and plots training loss after training.

"""

import pandas as pd
from datasets import Dataset, DatasetDict
import random
from transformers import GPT2Tokenizer, GPT2LMHeadModel, TrainingArguments, Trainer, DataCollatorForSeq2Seq, TrainerCallback
import torch
import matplotlib.pyplot as plt
import csv
from transformers import DataCollatorWithPadding
import re
import csv
import os
from tqdm import tqdm
from openai import OpenAI


positive_prompts = [
    "Write a positive movie review for the movie '{movie}' given the following summary:",
    "Praise the movie '{movie}'. Summary:",
    "Explain why '{movie}' was excellent. Summary:",
    "Describe why '{movie}' is a must-watch. Summary:",
    "Give a glowing review for '{movie}'. Summary:",
]

negative_prompts = [
    "Write a negative movie review for the movie '{movie}' given the following summary:",
    "Criticize the movie '{movie}'. Summary:",
    "Explain why '{movie}' was terrible. Summary:",
    "Describe why '{movie}' is disappointing. Summary:",
    "Give a harsh review for '{movie}'. Summary:",
]

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

def create_instruction_pair(example):
    """
    Create an instruction + output pair from a dataset example.

    Args:
        example (dict): A dictionary with keys 'movie_name', 'summary', 'review', and 'label'.

    Returns:
        dict: A dictionary with keys 'input' (instruction text) and 'output' (review text).
    """
    movie = example["movie_name"]
    summary = example["summary"]
    review = example["review"]

    if example["label"] == 1:
        prompt_template = random.choice(positive_prompts)
    else:
        prompt_template = random.choice(negative_prompts)

    instruction = prompt_template.format(movie=movie) + f"\n{summary}\n"

    return {
        "input": instruction.strip(),
        "output": review.strip()
    }


def preprocess_function(examples):
    """
    Tokenize dataset examples and prepare labels for instruction-tuning.

    Args:
        examples (dict): Dataset batch with keys 'input' and 'output'.

    Returns:
        dict: Tokenized inputs and masked labels for causal LM fine-tuning.
    """
    instructions = examples["input"]
    outputs = examples["output"]

    input_texts = [instruction + output for instruction, output in zip(instructions, outputs)]
    full_tokenized = tokenizer(
        input_texts,
        padding="max_length",
        max_length=512,
        truncation=True,
    )

    labels = []
    for instruction, output in zip(instructions, outputs):
        instr_ids = tokenizer(instruction, truncation=True, max_length=256).input_ids
        full_ids = tokenizer(instruction + output, truncation=True, max_length=512).input_ids

        label = [-100] * len(instr_ids) + full_ids[len(instr_ids):]
        if len(label) < 512:
            label += [-100] * (512 - len(label))
        else:
            label = label[:512]

        labels.append(label)

    full_tokenized["labels"] = labels
    return full_tokenized


def finetune(
    train_file,
    test_labels_reviews_file,
    test_labels_movies_file,
    test_summaries_file,
    model_name="gpt2",
    output_dir="./model/gpt2-instruction-finetuned",
    logging_dir="./instruction-finetuning-logs",
    num_train_epochs=3,
    batch_size=2,
    plot_save_file=None
):
    """
    Instruction-tune a GPT-2 model on a dataset of movie summaries and reviews.

    Args:
        train_file (str): Path to the training CSV file.
        test_labels_reviews_file (str): Path to test reviews CSV file.
        test_labels_movies_file (str): Path to test movie names CSV file.
        test_summaries_file (str): Path to test summaries CSV file.
        model_name (str): Hugging Face model name to load (default: 'gpt2').
        output_dir (str): Directory to save the fine-tuned model.
        logging_dir (str): Directory to save logs.
        num_train_epochs (int): Number of training epochs.
        batch_size (int): Batch size for training and evaluation.
        plot_save_file (str or None): If provided, saves the loss plot image to this file.

    Returns:
        None
    """

    train_df = pd.read_csv(
        train_file,
        engine="python",
        quoting=csv.QUOTE_ALL,
        quotechar='"',
        escapechar='\\',
        on_bad_lines='warn'  
    )

    test_labels_reviews_df = pd.read_csv(test_labels_reviews_file)
    test_labels_movienames_df = pd.read_csv(test_labels_movies_file)
    test_labels_movienames_summary_df= pd.read_csv(test_summaries_file)

    test_df = pd.concat([test_labels_movienames_summary_df, test_labels_reviews_df["text"]], axis=1)
    test_df.rename(columns={'text': 'review'}, inplace=True)
    print(f"Rows in train_df: {len(train_df)}")
    print(f"Rows in test_df: {len(test_df)}")


    # Apply function to datasets
    train_records = train_df.to_dict(orient="records")
    train_pairs = [create_instruction_pair(record) for record in train_records]

    test_records = test_df.to_dict(orient="records")
    test_pairs = [create_instruction_pair(record) for record in test_records]

    # Create HF Dataset
    train_dataset = Dataset.from_list(train_pairs)
    test_dataset = Dataset.from_list(test_pairs)

    # Create HF Dataset
    dataset_dict = DatasetDict({
        "train": train_dataset,
        "test": test_dataset
    })

    # Tokenizer and model
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.eos_token_id

    tokenized_datasets = dataset_dict.map(preprocess_function, batched=True, remove_columns=["input", "output"])


    # Data collator
    data_collator = DataCollatorWithPadding(tokenizer)


    # Custom callback for logging losses
    class LoggingCallback(TrainerCallback):
        def __init__(self):
            self.logs = []

        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs is not None and "loss" in logs:
                self.logs.append((state.global_step, logs["loss"]))

    logging_callback = LoggingCallback()

    # Training arguments
    training_args = TrainingArguments(
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        save_steps=500,
        save_total_limit=2,
        save_strategy="epoch",
        eval_steps=500,
        eval_strategy="steps",  
        prediction_loss_only=True,
        fp16=torch.cuda.is_available(),

        output_dir=output_dir,
        overwrite_output_dir=True,
        logging_steps=100,
        logging_dir=logging_dir, 
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[logging_callback]
    )


    trainer.train()


    # Save model
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Plot training loss
    if logging_callback.logs:
        steps, losses = zip(*logging_callback.logs)
        plt.figure()
        plt.title("Instruction Finetuning - Training Loss")
        plt.plot(steps, losses)
        plt.xlabel("steps")
        plt.ylabel("loss")
        if plot_save_file:
            plt.savefig(plot_save_file)
            print(f"Plot saved to {plot_save_file}")
        else:
            plt.show()
    else:
        print("No logs recorded")

    print("Saved model instruction-finetuning!")


def run_movie_extraction_pipeline(
    imdb_test_path="data/imdb_test.csv",
    labels_and_movies_path="data/imdb_test_labels_and_movies.csv",
    summaries_path="data/imdb_test_with_summaries.csv",
    cleaned_labels_and_movies_path="data/cleaned_labels_and_movies.csv",
    cleaned_reviews_path="data/cleaned_imdb_test.csv",
    cleaned_summaries_path="data/cleaned_imdb_test_with_summaries.csv",
    openai_api_key="YOUR-API-KEY"
):

    client = OpenAI(api_key=openai_api_key)

    # --- Step 1: Extract movie names ---

    df = pd.read_csv(imdb_test_path)

    with open(labels_and_movies_path, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["label", "movie_name"])

    def extract_movie_name(review_text):
        try:
            system_msg = (
                "You are an expert at identifying the main movie being reviewed. "
                "Return the exact name of the movie being reviewed. "
                "If multiple movies are mentioned, choose the one being criticized or evaluated. "
                "Always try to make an educated guess. Do NOT return 'UNKNOWN' unless impossible."
            )
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": f"Review: {review_text}"}
                ],
                temperature=0,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"[Movie Name Error] {e}")
            return "ERROR"

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting Movie Names"):
        movie = extract_movie_name(row["text"])
        with open(labels_and_movies_path, "a", newline='') as f:
            writer = csv.writer(f)
            writer.writerow([row["label"], movie])
            f.flush()
            os.fsync(f.fileno())

    # --- Step 2: Generate summaries ---

    df_movies = pd.read_csv(labels_and_movies_path)

    with open(summaries_path, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["label", "movie_name", "summary"])

    def generate_summary(movie_name):
        try:
            if movie_name.upper() in ["ERROR", "UNKNOWN", ""]:
                return "Could not summarize"

            system_msg = (
                "You are a helpful movie assistant. Given a movie name, write a one-sentence plot summary. "
                "If invalid, return 'Could not summarize'."
            )
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": f"Movie title: {movie_name}"}
                ],
                temperature=0,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"[Summary Error] {movie_name} → {e}")
            return "Could not summarize"

    for _, row in tqdm(df_movies.iterrows(), total=len(df_movies), desc="Generating Summaries"):
        summary = generate_summary(row["movie_name"])
        with open(summaries_path, "a", newline='') as f:
            writer = csv.writer(f)
            writer.writerow([row["label"], row["movie_name"], summary])
            f.flush()
            os.fsync(f.fileno())

    # --- Step 3: Clean dataset ---

    df_movies = pd.read_csv(labels_and_movies_path)
    df_reviews = pd.read_csv(imdb_test_path)
    df_summaries = pd.read_csv(summaries_path)

    vague_phrases = [
        "without more specific details", "difficult to determine", "not enough specific information",
        "exact title is not provided", "not mentioned", "could be any", "likely an adult film",
        "likely a full moon production", "focuses on the production quality", "tone suggests",
        "widely criticized", "major disappointment", "low-budget", "challenging to determine",
        "if you have any additional details", "head of the family is suggested", "does not provide enough",
        "generic and does not provide enough", "unclear", "error", "unknown", "not a movie",
        "tv series", "television episode", "tv episode", "episode from the", "the review does not provide explicit details",
        "based on the review provided", "the review is very general", "it sounds like the review is for a movie",
        "the review seems to be critiquing the film", "the review is about an episode", "the review is likely referring to",
        "the review is likely about"
    ]

    def is_vague(name):
        if not isinstance(name, str):
            return True
        name_lower = name.lower().strip()
        return any(phrase in name_lower for phrase in vague_phrases) or "the review" in name_lower

    def extract_from_phrase(text):
        if not isinstance(text, str):
            return text
        patterns = [
            r'[Tt]he movie being reviewed (is|is likely) ["“\']([^"”\']+)["”\']',
            r'[Tt]he review is (likely )?about the movie ["“\']([^"”\']+)["”\']',
            r'[Tt]he review is likely referring to ["“\']([^"”\']+)["”\']',
            r'[Tt]he review seems to be critiquing the film ["“\']([^"”\']+)["”\']',
            r'[Tt]he film being reviewed is likely ["“\']([^"”\']+)["”\']'
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(len(match.groups())).strip()
        return text.strip()

    invalid_name_indices = df_movies[df_movies["movie_name"].apply(is_vague)].index
    df_movies_clean = df_movies.drop(index=invalid_name_indices).reset_index(drop=True)
    df_reviews_clean = df_reviews.drop(index=invalid_name_indices).reset_index(drop=True)
    df_summaries_clean = df_summaries.drop(index=invalid_name_indices).reset_index(drop=True)

    df_movies_clean["movie_name"] = df_movies_clean["movie_name"].apply(extract_from_phrase)
    df_summaries_clean["movie_name"] = df_summaries_clean["movie_name"].apply(extract_from_phrase)

    invalid_summary_indices = df_summaries_clean[
        df_summaries_clean["summary"].astype(str).str.strip().str.lower().str.startswith("could not summarize")
    ].index

    df_movies_clean = df_movies_clean.drop(index=invalid_summary_indices).reset_index(drop=True)
    df_reviews_clean = df_reviews_clean.drop(index=invalid_summary_indices).reset_index(drop=True)
    df_summaries_clean = df_summaries_clean.drop(index=invalid_summary_indices).reset_index(drop=True)

    df_movies_clean.to_csv(cleaned_labels_and_movies_path, index=False)
    df_reviews_clean.to_csv(cleaned_reviews_path, index=False)
    df_summaries_clean.to_csv(cleaned_summaries_path, index=False)

    print("Cleaned datasets saved.")

# Example usage (you can replace with your paths and key)
# run_movie_extraction_pipeline(openai_api_key="your-openai-api-key")

