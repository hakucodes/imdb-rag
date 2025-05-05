import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load the dataset
df = pd.read_csv('./data/IMDB Dataset.csv')

# Initialize text splitter for chunking long reviews
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,  # Max characters per chunk
    chunk_overlap=50  # Overlap to retain context
)

# Function to split reviews into chunks
def split_reviews(text):
    return text_splitter.split_text(text)

# Apply splitting and create a list of documents
documents = []
for idx, row in df.iterrows():
    chunks = split_reviews(row['review'])
    for chunk in chunks:
        documents.append({
            'text': chunk,
            'sentiment': row['sentiment'],
            'id': f'review_{idx}_{len(chunks)}'
        })

# Save processed documents for inspection (optional)
import json
with open('./data/processed_reviews.json', 'w') as f:
    json.dump(documents, f, indent=2)

print(f"Processed {len(documents)} document chunks.")