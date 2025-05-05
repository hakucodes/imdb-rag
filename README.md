# IMDB Movie Reviews Sentiment Analysis Pipeline

## Prerequisites
- Python environment with `uv` package manager installed
- Internet connection to download the dataset

## Dataset Setup
1. Download the IMDB dataset from Kaggle:
    - [IMDB Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews?resource=download)
    - You'll need a Kaggle account to download the dataset

2. Place the downloaded dataset in the `data` folder of your project

## Running the Pipeline

### 1. Preprocess the Data
Execute the preprocessing script:
```shell
uv run preprocess_imdb.py
```
This script prepares the IMDB reviews for analysis.

### 2. Run the RAG Pipeline
Execute the RAG (Retrieval-Augmented Generation) pipeline:
```shell
uv run rag_pipeline.py
```
This will process the preprocessed data using the RAG architecture.

## Note
Make sure all dependencies are installed before running the scripts.
