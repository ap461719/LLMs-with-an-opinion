# inference_eval.py

"""
inference_eval.py

This module performs inference and evaluation of fine-tuned or pretrained language models
(GPT-2 and LLaMA-2-7B) for sentiment-conditioned movie review generation tasks. It loads 
models using Hugging Face Transformers and PEFT, constructs prompts based on sentiment labels 
and movie metadata, and generates reviews using sampling-based decoding.

The generated outputs are evaluated against ground-truth IMDb reviews using a suite of metrics:
- BLEU and ROUGE-L for lexical overlap
- Sentence-BERT cosine similarity for semantic alignment
- BERT-based sentiment classification for sentiment match accuracy

Usage:
    - Load a model with `load_model()`
    - Call `evaluate_model()` with the test dataframe and decoding parameters
"""

import torch
import pandas as pd
import csv
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    AutoModelForSequenceClassification,
    BitsAndBytesConfig, GPT2Tokenizer
)
from peft import PeftModel
from utils.inference_eval_utils import (
    generate_reviews, get_cls_embeddings, classify_sentiments,
    compute_bleu_scores, compute_cosine_similarity, compute_rouge_scores
)

# --- Configuration ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# General Config
bert_model_name = "textattack/bert-base-uncased-imdb"
batch_size = 16

# Load Sentiment Model
bert_tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
bert_model = AutoModelForSequenceClassification.from_pretrained(bert_model_name).to(device)
bert_model.eval()


# --- Model Loader Function ---
def load_model(model_type="gpt2-summaries", pretrained=False):
    if model_type == "llama":
        llama_base_model = "meta-llama/Llama-2-7b-hf"
        tokenizer = AutoTokenizer.from_pretrained(llama_base_model, padding_side='left')
        tokenizer.pad_token = tokenizer.eos_token

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, 
            llm_int8_enable_fp32_cpu_offload=True 
        )

        base_model = AutoModelForCausalLM.from_pretrained(
            llama_base_model, quantization_config=bnb_config, device_map="auto"
        )

        if pretrained: 
            model = base_model
        else:
            peft_model_path = "./models/llama-7b-qlora-finetuned"
            model = PeftModel.from_pretrained(base_model, peft_model_path).to(device)


    elif model_type == "gpt2":
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2", padding_side="left")
        tokenizer.pad_token = tokenizer.eos_token

        if pretrained:
            model = AutoModelForCausalLM.from_pretrained("gpt2").to(device)
        else:
            gpt_model_path = "./models/gpt2-finetuned-augmented/checkpoint-26250/"
            model = AutoModelForCausalLM.from_pretrained(gpt_model_path).to(device)

    elif model_type == 'gpt2-summaries': 
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2", padding_side="left")
        tokenizer.pad_token = tokenizer.eos_token

        if pretrained:
            model = AutoModelForCausalLM.from_pretrained("gpt2").to(device)
        else:
            gpt_model_path = "./models/gpt2-instruction-finetuned"
            model = AutoModelForCausalLM.from_pretrained(gpt_model_path).to(device)
        
    else:
        raise ValueError("Unsupported model type")

    model.eval()
    return model, tokenizer

def evaluate_model(model_and_tokenizer, df, prompt_type, top_k=50, top_p=0.95, temperature=0.7, batch_size=16):
    model, tokenizer = model_and_tokenizer

    bleu_scores, rouge_scores, cos_sims, sentiments = [], [], [], []
    generated_examples = []

    for i in tqdm(range(0, len(df), batch_size)):
        batch = df.iloc[i:i+batch_size]
        

        if prompt_type == 'label-moviename':
            prompts = [f"Give me a {'positive' if lbl == 1 else 'negative'} review for the movie {name}."
                       for lbl, name in zip(batch["label"], batch["movie_name"])]

        elif prompt_type == 'label-moviename-summary':
            prompts = [f"Give me a {'positive' if lbl == 1 else 'negative'} review for the movie {name}. This is a short summary of the movie: {summary}"
                       for lbl, name, summary in zip(batch["label"], batch["movie_name"], batch["summary"])]

        references = batch["review"].tolist()

        generated_reviews = generate_reviews(model, tokenizer, prompts, device,
                                             top_k=top_k, top_p=top_p, temperature=temperature)

        # Strip the prompt part from the generated reviews
        stripped_reviews = [review[len(prompt):].strip() for review, prompt in zip(generated_reviews, prompts)]

        if len(generated_examples) < 30: 
            generated_examples.extend(stripped_reviews[:30 - len(generated_examples)])

        bleu_scores += compute_bleu_scores(references, stripped_reviews)
        rouge_result = compute_rouge_scores(references, stripped_reviews)
        rouge_scores.append(rouge_result["rougeL"])

        emb_ref = get_cls_embeddings(references, bert_model, bert_tokenizer, device)
        emb_gen = get_cls_embeddings(stripped_reviews, bert_model, bert_tokenizer, device)
        cos_sims += compute_cosine_similarity(emb_ref, emb_gen)

        predicted_sentiments = classify_sentiments(stripped_reviews, bert_model, bert_tokenizer, device)
        expected_sentiments = ["Positive" if l == 1 else "Negative" for l in batch["label"]]
        sentiments += [int(pred == exp) for pred, exp in zip(predicted_sentiments, expected_sentiments)]

    movie_names = df["movie_name"].head(30).tolist()
    labels = df["label"].head(30).tolist()

    print("\n--- First 30 Generated Examples ---")
    for idx, (l, mv_n, example) in enumerate(zip(labels, movie_names, generated_examples), 1):
        sentiment = "positive" if l == 1 else "negative"
        print(f"{idx}. Movie: {mv_n}\n   Sentiment: {sentiment}\n   Generated Review: {example}\n")

    average_bleu = round(sum(bleu_scores)/len(bleu_scores), 4)
    average_rouge = round(sum(rouge_scores)/len(rouge_scores), 4)
    average_semantic_similarity = round(sum(cos_sims)/len(cos_sims), 4)
    sentiment_match_accuracy = round(sum(sentiments)/len(sentiments), 4)

    # --- Results Summary ---
    print(f"\n--- Evaluation Summary (Prompt type {prompt_type}) ---")
    print(f"Avg BLEU: {average_bleu}")
    print(f"Avg ROUGE-L: {average_rouge}")
    print(f"Avg Semantic Similarity: {average_semantic_similarity}")
    print(f"Sentiment Match Accuracy: {sentiment_match_accuracy}%")

    return average_bleu, average_rouge, average_semantic_similarity, sentiment_match_accuracy
