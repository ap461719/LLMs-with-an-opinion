"""
BERT_utils.py

This module provides utility functions for performing sentiment analysis and computing
text similarity using pretrained BERT models from Hugging Face. It includes functions
for both single-example and batched sentiment classification using models fine-tuned
on the IMDB and product review datasets. It also provides a utility for measuring 
cosine similarity between [CLS] embeddings of two texts.

Available functions:
- classify_sentiment_bert_imdb: Classifies a single movie review as positive or negative.
- classify_sentiment_bert_imdb_batch: Classifies a batch of movie reviews and returns predictions and confidence scores.
- bert_cls_cosine_similarity: Computes cosine similarity between two texts using BERT [CLS] embeddings.
- classify_sentiment_bert_products: Classifies a single product review into a 1-5 star rating with confidence.

"""

import torch.nn.functional as F
import torch

def classify_sentiment_bert_imdb(model, tokenizer, review_text):
    """
    Classify sentiment of a given movie review using the textattack/bert-base-uncased-imdb BERT model.

    This function classifies a single movie review as either positive or negative sentiment.
    The model was fine-tuned on the IMDB dataset for binary sentiment classification.

    Args:
        model (transformers.PreTrainedModel): Pretrained BERT model fine-tuned on the IMDB dataset.
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer corresponding to the BERT model.
        review_text (str): The review text to classify.

    Returns:
        Tuple[str, float]:
            - The predicted sentiment label ("positive" or "negative").
            - The confidence score for the predicted sentiment (float value between 0 and 1).
    """
    id2label = {0: "negative", 1: "positive"}
    
    inputs = tokenizer(review_text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1)
        pred_id = torch.argmax(probs, dim=1).item()
        return id2label[pred_id], probs[0][pred_id].item()
    
def classify_sentiment_bert_imdb_batch(model, tokenizer, review_texts, device="cpu"):
    """
    Classify sentiment of a batch of movie reviews using the textattack/bert-base-uncased-imdb model.

    This function takes a list of review texts and returns sentiment predictions in batched form.
    It uses the BERT model fine-tuned on the IMDB dataset to predict whether each review is
    positive (1) or negative (0). It also returns the model's confidence for each prediction.

    Args:
        model (transformers.PreTrainedModel): Pretrained BERT model for sequence classification.
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer corresponding to the BERT model.
        review_texts (List[str]): List of review strings to classify.
        device (str): The device to run inference on ("cpu" or "cuda").

    Returns:
        Tuple[List[int], List[float]]:
            - List of predicted labels (0 for negative, 1 for positive).
            - List of confidence scores corresponding to the predicted class for each review.
    """
    inputs = tokenizer(review_texts, return_tensors="pt", truncation=True, padding=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1)
        pred_ids = torch.argmax(probs, dim=1)
        pred_labels = [i.item() for i in pred_ids]
        confidences = [probs[i, pred_ids[i]].item() for i in range(len(pred_ids))]
    return pred_labels, confidences


def bert_cls_cosine_similarity(text1, text2, model, tokenizer, device="cpu"):
    """
    Compute cosine similarity between the [CLS] embeddings of two input texts
    using a BERT model (e.g., textattack/bert-base-uncased-imdb).

    Args:
        text1 (str): First input text.
        text2 (str): Second input text.
        model (transformers.PreTrainedModel): BERT model (use AutoModel, not classification head).
        tokenizer (transformers.PreTrainedTokenizer): Corresponding tokenizer.
        device (str): Device to run inference on ("cpu" or "cuda").

    Returns:
        float: Cosine similarity between the two [CLS] embeddings (range: -1 to 1).
    """
    def get_cls_embedding(text):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            return outputs.last_hidden_state[:, 0, :].squeeze()

    model.to(device)
    model.eval()

    emb1 = get_cls_embedding(text1)
    emb2 = get_cls_embedding(text2)

    similarity = F.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)).item()
    return similarity



def classify_sentiment_bert_products(model, tokenizer, review_text):
    """
    Classify sentiment of a given product review using the nlptown/bert-base-multilingual-uncased-sentiment BERT model.

    This function classifies a single product review into one of five sentiment categories, 
    ranging from 1 star to 5 stars. The model was fine-tuned on product reviews for multi-class sentiment analysis.

    Args:
        model (transformers.PreTrainedModel): Pretrained BERT model fine-tuned on product reviews.
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer corresponding to the BERT model.
        review_text (str): The product review text to classify.

    Returns:
        Tuple[str, float]:
            - The predicted sentiment label (e.g., "1 star", "2 stars", "3 stars", "4 stars", "5 stars").
            - The confidence score for the predicted sentiment (float value between 0 and 1).
    """
    id2label = {
        0: "1 star",
        1: "2 stars",
        2: "3 stars",
        3: "4 stars",
        4: "5 stars"
    }
    
    inputs = tokenizer(review_text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1)
        pred_id = torch.argmax(probs, dim=1).item()
        return id2label[pred_id], probs[0][pred_id].item()


