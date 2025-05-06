from pinecone import Pinecone
from openai import OpenAI
import json
from tqdm import tqdm
from config import settings

# Initialize OpenAI client
openai_client = OpenAI(api_key=settings.OPENAI.API_KEY.get_secret_value())

# Initialize Pinecone
pc = Pinecone(
    api_key=settings.PINECONE.API_KEY.get_secret_value(),
)

# Create or connect to Pinecone index
index_name = "imdb-reviews"
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric="cosine",
        spec={"serverless": {"cloud": "aws", "region": "us-east-1"}},
    )
index = pc.Index(index_name)

# Load processed documents
with open("./data/processed_reviews.json", "r") as f:
    documents = json.load(f)

# Generate embeddings and store in Pinecone
batch_size = 100
for i in tqdm(range(0, len(documents), batch_size), desc="Embedding and upserting"):
    batch = documents[i : i + batch_size]
    texts = [doc["text"] for doc in batch]
    ids = [doc["id"] for doc in batch]
    metadata = [{"text": doc["text"], "sentiment": doc["sentiment"]} for doc in batch]

    # Generate embeddings
    response = openai_client.embeddings.create(
        model=settings.OPENAI.EMBEDDING_MODEL, input=texts
    )
    embeddings = [r.embedding for r in response.data]

    # Upsert to Pinecone
    to_upsert = list(zip(ids, embeddings, metadata))
    index.upsert(vectors=to_upsert)

print(f"Stored {len(documents)} vectors in Pinecone index '{index_name}'.")
