from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone as PineconeClient
from openai import OpenAI
from config import settings

# Initialize OpenAI client
openai_client = OpenAI(api_key=settings.OPENAI.API_KEY.get_secret_value())

# Initialize Pinecone
pc = PineconeClient(
    api_key=settings.PINECONE.API_KEY.get_secret_value(),
)

# Initialize Pinecone vector store
embeddings = OpenAIEmbeddings(model='text-embedding-ada-002', api_key=settings.OPENAI.API_KEY)
vector_store = PineconeVectorStore(index_name='imdb-reviews', embedding=embeddings, pinecone_api_key=settings.PINECONE.API_KEY.get_secret_value())

# Function to perform RAG
def run_rag(query, k=3):
    # Retrieve relevant documents
    docs = vector_store.similarity_search(query, k=k)
    
    context = "\n".join([doc.page_content for doc in docs])

    # Craft prompt for ChatGPT
    prompt = f"""You are a helpful assistant. Use the following context to answer the query.
    If the context doesn't provide enough information, say so and provide a general answer.
    Context:
    {context}

    Query: {query}
    Answer:"""

    # Call ChatGPT API
    response = openai_client.chat.completions.create(
        model=settings.OPENAI.MODEL,
        messages=[
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {'role': 'user', 'content': prompt}
        ],
        max_tokens=500,
        temperature=0.7
    )

    return response.choices[0].message.content

# Example usage
if __name__ == "__main__":
    print("Welcome to the RAG Movie Review Assistant!")
    print("You can ask questions about movie reviews.")
    query = input("Ask a question about a movie review: ")
    answer = run_rag(query)
    print(f"Answer: {answer}")