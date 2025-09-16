# inference_eval_utils.py

"""
inference_eval_utils.py

This utility module provides core functions for running inference and evaluation
on generated movie reviews from fine-tuned LLMs (e.g., GPT-2, LLaMA). It includes:

- `generate_reviews`: Sample-based text generation using top-k, top-p, and temperature.
- `get_cls_embeddings`: Extract [CLS] embeddings from a BERT model for semantic comparison.
- `classify_sentiments`: Perform binary sentiment classification using a fine-tuned BERT model.
- `compute_bleu_scores`: Compute BLEU scores for n-gram lexical overlap.
- `compute_rouge_scores`: Compute ROUGE-L scores using the `evaluate` library.
- `compute_cosine_similarity`: Measure semantic similarity between BERT embeddings.

These functions are used within `inference_eval.py` to perform both quantitative and qualitative
analysis of generated outputs.
"""

import torch
import torch.nn.functional as F
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import evaluate


smooth_fn = SmoothingFunction().method1
rouge_metric = evaluate.load('rouge')


# Generate Reviews
def generate_reviews(model, tokenizer, prompts, device, max_length=200, top_k=50, top_p=0.95, temperature=0.7):

   torch.cuda.empty_cache()


   inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)
   with torch.no_grad():
       outputs = model.generate(
           **inputs,
           max_length=max_length,
           do_sample=True,
           top_k=top_k,
           top_p=top_p,
           temperature=temperature,
           pad_token_id=tokenizer.eos_token_id
       )
   return tokenizer.batch_decode(outputs, skip_special_tokens=True)


# Semantic Embeddings
def get_cls_embeddings(texts, bert_model, bert_tokenizer, device):
   inputs = bert_tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(device)
   with torch.no_grad():
       outputs = bert_model.bert(**inputs)
       return outputs.last_hidden_state[:, 0, :]


# Sentiment Classifier
def classify_sentiments(texts, bert_model, bert_tokenizer, device):
   inputs = bert_tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(device)
   with torch.no_grad():
       outputs = bert_model(**inputs)
       preds = torch.argmax(F.softmax(outputs.logits, dim=-1), dim=-1)
       return ["Positive" if p == 1 else "Negative" for p in preds]


# Compute BLEU scores
def compute_bleu_scores(references, hypotheses):
   return [sentence_bleu([ref.split()], hyp.split(), smoothing_function=smooth_fn)
           for ref, hyp in zip(references, hypotheses)]


# Compute ROUGE scores
def compute_rouge_scores(references, hypotheses):
   results = rouge_metric.compute(predictions=hypotheses, references=references)
   return results


# Compute Cosine Similarity
def compute_cosine_similarity(emb_ref, emb_gen):
   return F.cosine_similarity(emb_ref, emb_gen, dim=1).tolist()



