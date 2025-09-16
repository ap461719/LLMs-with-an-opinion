[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/Yd55b8hB)
# E6691 Spring 2025: Final Project

# LLMs with an Opinion: Controlling and Evaluating Sentiment in GPT-2 Generated Movie Reviews 

### Group Members
- Sri Iyengar (si2468)
- Anushka Pachuary (ap4617)
- Radhika Patel (rpp2142)


## Description

This project investigates sentiment-aligned movie review generation using large language models (LLMs), specifically GPT-2 and LLaMA-2-7B. The goal is to generate coherent and contextually accurate movie reviews conditioned on user-specified sentiment (positive/negative), movie names, and summaries.

We curated a novel instruction-tuning dataset from the IMDb corpus by extracting movie names and generating one-sentence summaries using the OpenAI GPT-4 API. GPT-2 was fine-tuned on instruction-summary-review triplets using masked loss, while LLaMA-2 was fine-tuned using QLoRA adapters on open-ended sentiment prompts. Evaluation was performed on both pretrained and fine-tuned models across multiple decoding strategies (top-k, top-p, temperature) and prompt styles.



## Key Components
- Data Curation: IMDb reviews augmented with movie names and GPT-generated summaries.

- Instruction Tuning:

  - GPT-2: Full-parameter fine-tuning with masked loss.

  - LLaMA-2: 4-bit quantized model fine-tuned with LoRA adapters via PEFT.

- Evaluation: BLEU, ROUGE-L, cosine similarity (Sentence-BERT), and BERT-based sentiment classification.

- Tech Stack: Hugging Face Transformers, datasets, bitsandbytes, peft, OpenAI API, PyTorch, TensorBoard.

## Results

GPT-2: Achieved up to 98% sentiment accuracy post fine-tuning.

LLaMA-2: Reached 96.7% sentiment accuracy using 4-bit QLoRA.


## Directory Structure

```
├── data
│   ├── cleaned_imdb_test_with_summaries.csv # final cleaned movie_name, summary, review, test dataset
│   ├── cleaned_imdb_test.csv
│   ├── cleaned_imdb_train_25k.csv                                  
│   ├── cleaned_imdb_train.csv               # final cleaned movie_name, summary, review, train dataset
│   ├── cleaned_labels_and_movies.csv
│   ├── description.txt
│   ├── imdb_test_with_prompt.csv
│   ├── imdb_test.csv                        # original hugging face test dataset (label, review)
│   ├── imdb_train_with_prompt.csv           
│   └── imdb_train.csv                       # original hugging face train dataset (label, review)
├── E6691.2025Spring.IDLI.report.si2468.ap4617.rpp2142.pdf
├── logs                                     # tensorboard logs to visualize llama-finetuning
│   ├── events.out.tfevents.1746399878.instance-l4
│   └── events.out.tfevents.1746400312.instance-l4
├── models
│   ├── description.txt
├── utils
│   ├── BERT_utils.py                        # additional functions for BERT evaluation 
│   ├── gpt_fine_tuning.py                   # function for instruction-finetuning gpt2 and curating summaries dataset to train it on
│   ├── inference_eval_utils.py              # functions for metrics 
│   ├── inference_eval.py                    # helper function for loading model and building inferencing pipeline for single experiment  # functions for instruction-finetuning gpt2
│   └── llama_fine_tuning.py                 # function for instruction-finetuning gpt2 and augmenting dataset to train it on 
├── .gitignore
├── main.ipynb                               # main jupyter driver notebook
├── README.md
└── requirements.txt


```

## Link to Google Drive

Both finetuned LLaMA-QLoRa layers and the our GPT-2 instruction-finetuned model can be found [here](https://drive.google.com/drive/folders/1mFPbbiAUrmcHWo_JkNGFzcqQSbIshJVh). Simply unzip them under the models/ folder. 

## Visualizing Tensorboard Logs

```bash
tensorboard --logdir=./logs --port=6006
```

## Setting Up Environment

```bash
pip install virtualenv
virtualenv venv
source venv/bin/activate
pip install -r requirements.txt
```