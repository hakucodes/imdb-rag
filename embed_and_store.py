import os
import pinecone
from openai import OpenAI
import json
from tqdm import tqdm

# Initialize OpenAI client
openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Initialize Pinecone
pinecone.init(
    api_key=os.getenv('PINECONE_API_KEY'),
    environment=os.getenv('PINECONE_ENVIRONMENT')
)

# Create or connect to Pinecone index
index_name = 'imdb-reviews'
if index_name not in pinecone.list_indexes():
    pinecone.create_index(index_name, dimension=1536, metric='cosine')
index = pinecone.Index(index_name)

# Load processed documents
with open('./data/processed_reviews.json', 'r') as f:
    documents = json.load(f)

# Generate embeddings and store in Pinecone
batch_size = 100
for i in tqdm(range(0, len(documents), batch_size), desc="Embedding and upserting"):
    batch = documents[i:i + batch_size]
    texts = [doc['text'] for doc in batch]
    ids = [doc['id'] for doc in batch]
    metadata = [{'text': doc['text'], 'sentiment': doc['sentiment']} for doc in batch]

    # Generate embeddings
    response = openai_client.embeddings.create(
        model='text-embedding-ada-002',
        input=texts
    )
    embeddings = [r.embedding for r in response.data]

    # Upsert to Pinecone
    to_upsert = list(zip(ids, embeddings, metadata))
    index.upsert(vectors=to_upsert)

print(f"Stored {len(documents)} vectors in Pinecone index '{index_name}'.")