# MLOps Assignment 2 — DistilBERT Goodreads Genre Classifier

**IIT Jodhpur | PGD AI Programme | MLOps**

Fine-tunes `distilbert-base-cased` on UCSD Goodreads reviews to classify books into 7 genres.
Experiment tracking with Weights & Biases and model hosting on Hugging Face Hub.

## Project Description

This project implements a complete MLOps workflow for text classification. A pre-trained DistilBERT model is fine-tuned on Goodreads book reviews from the UCSD Book Graph dataset to predict book genres (poetry, comics & graphic, fantasy & paranormal, history & biography, mystery/thriller/crime, romance, young adult). The entire training pipeline is modularised into clean Python scripts, tracked with W&B, and the final model is published to the Hugging Face Hub.

## Repository Structure

```text
├── data.py           # Data loading, sampling, train/test split, tokenizer encoding
├── train.py          # Model loading, Trainer setup, W&B logging, HF Hub push
├── eval.py           # Final evaluation, classification report, W&B Artifact upload
├── utils.py          # Label maps, MyDataset class, compute_metrics
├── requirements.txt  # Python dependencies
└── README.md

