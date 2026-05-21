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

Setup Instructions1. Clone the repositoryBashgit clone [https://github.com/g25ait2004-cyber/Abhishek.git](https://github.com/g25ait2004-cyber/Abhishek.git)
cd Abhishek
2. Install dependenciesBashpip install -r requirements.txt
3. Set environment variablesBashexport WANDB_API_KEY=<your_wandb_api_key>
export HF_TOKEN=<your_huggingface_token>
4. Run trainingBashpython train.py
5. Run evaluationBashpython eval.py
GPU note: Run on Google Colab or Kaggle for a GPU. On CPU, reduce reviews_per_genre in data.py to 200 to keep training time manageable.ResultsBased on the latest evaluation run (zvs34qon):MetricScoreAccuracy0.5750F1 Score0.5729Eval Loss2.2505LinksHugging Face model: g25ait2004/DistilBERT_GoodreadsW&B dashboard: MLOps Assignment Project
