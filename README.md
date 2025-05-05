# IMDB Movie Reviews Sentiment Analysis Pipeline

## Prerequisites
- Python environment with `uv` package manager installed
- Internet connection to download the dataset
- OpenAI API key
- Pinecone API key

## Project Structure Setup

1. Download the IMDB dataset from Kaggle:
    - [IMDB Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews?resource=download)
    - You'll need a Kaggle account to download the dataset
    - Place the downloaded `IMDB Dataset.csv` file in the `data` folder

2. Configure API Keys:
    - Rename `.env.example` to `.env`
    - Fill in your API keys:
    ```
    OPENAI_API_KEY=your_openai_key
    OPENAI_MODEL=your_preferred_model
    PINECONE_KEY=your_pinecone_key
    ```

Your project structure should look like this:
```
.
├── config.py
├── data
│   └── IMDB Dataset.csv
├── embed_and_store.py
├── preprocess_imdb.py
├── main.py
├── .env
└── README.md
```

## Running the Pipeline

### 1. Preprocess the Data
Execute the preprocessing script:
```shell
uv run preprocess_imdb.py
```
This script prepares the IMDB reviews for analysis and creates `processed_reviews.json` in the data folder.

### 2. Vectorize the Data
Execute the embedding script:
```shell
uv run embed_and_store.py
```
This script converts the preprocessed reviews into vectors and stores them in Pinecone.

### 3. Run the RAG Pipeline
Execute the RAG (Retrieval-Augmented Generation) pipeline:
```shell
uv run main.py
```
This will process the vectorized data using the RAG architecture.

## Note
Make sure all dependencies are installed and API keys are properly configured before running the scripts.

