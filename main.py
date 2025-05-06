import json
from pydantic import BaseModel, Field
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone as PineconeClient
from openai import OpenAI, OpenAIError
from config import settings
import logging

# Set up logging for better error tracking
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Pydantic model for structured response
class RagResponse(BaseModel):
    answer: str = Field(description="The generated answer to the query")
    confidence: float = Field(
        description="Confidence score based on document relevance (0 to 1)",
        ge=0.0,
        le=1.0,
    )
    sources: list[dict] = Field(
        description="list of source documents with text, sentiment, and score"
    )
    thought_process: list[str] = Field(
        default_factory=list,
        description="list of reasoning steps the LLM followed to answer the query"
    )


# Configuration for RAG parameters
RAG_CONFIG = {
    "k": 3,  # Number of documents to retrieve
    "max_tokens": 500,  # Max tokens for ChatGPT response
    "temperature": 0.7,  # Controls response creativity
    "model": settings.OPENAI.DEFAULT_MODEL,
}

# Initialize OpenAI client
openai_client = OpenAI(api_key=settings.OPENAI.API_KEY.get_secret_value())

# Initialize Pinecone
pc = PineconeClient(api_key=settings.PINECONE.API_KEY.get_secret_value())

# Initialize Pinecone vector store
embeddings = OpenAIEmbeddings(
    model=settings.OPENAI.EMBEDDING_MODEL, api_key=settings.OPENAI.API_KEY
)
vector_store = PineconeVectorStore(
    index_name="imdb-reviews",
    embedding=embeddings,
    pinecone_api_key=settings.PINECONE.API_KEY.get_secret_value(),
)


# Function to perform RAG
def run_rag(query: str, k: int = RAG_CONFIG["k"]) -> RagResponse:
    """
    Perform Retrieval-Augmented Generation with structured output.

    Args:
        query: User query string
        k: Number of documents to retrieve

    Returns:
        RagResponse: Structured response with answer, confidence, and sources
    """
    if not query.strip():
        logger.error("Empty query provided")
        raise ValueError("Query cannot be empty")

    try:
        # Retrieve relevant documents with scores
        docs_with_scores = vector_store.similarity_search_with_score(query, k=k)

        # Extract context and metadata
        context = []
        sources = []
        for doc, score in docs_with_scores:
            text = doc.page_content
            sentiment = doc.metadata.get("sentiment", "unknown")
            context.append(text)
            sources.append(
                {"text": text, "sentiment": sentiment, "score": round(score, 4)}
            )

        context_str = "\n\n".join(context)

        # Improved prompt to reduce hallucinations
        prompt = f"""You are a precise and factual assistant specializing in movie review analysis. Your task is to answer the query based solely on the provided context from IMDB reviews. Follow these rules:
        - Use only the information in the context to form your answer.
        - Do not speculate, use external knowledge, or make assumptions beyond the context.
        - If the context lacks sufficient information to answer fully, clearly state: "The provided reviews do not contain enough information to answer this query fully," and provide a brief general response if applicable.
        - Keep the answer concise, relevant, and focused on the query.
        - If summarizing sentiments, indicate the number of positive/negative reviews in the context.

        Provide your response in the following JSON format:
        {{
        "answer": "...",
        "thought_process": [
            "Step 1: ...",
            "Step 2: ...",
            ...
        ]
        }}

        Context:
        {context_str}

        Query: {query}
        """

        # Call ChatGPT API with error handling
        try:
            response = openai_client.chat.completions.create(
                model=RAG_CONFIG["model"],
                messages=[
                    {
                        "role": "system",
                        "content": "You are a precise movie review assistant.",
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=RAG_CONFIG["max_tokens"],
                temperature=RAG_CONFIG["temperature"],
            )
            # Inside run_rag
            response_text = (
                response.choices[0].message.content.strip()
                if response.choices[0].message.content
                else ""
            )

            # Attempt to parse JSON output from model
            try:
                structured_output = json.loads(response_text)
                answer = structured_output.get("answer", "")
                thought_process = structured_output.get("thought_process", [])
            except json.JSONDecodeError:
                logger.warning("Failed to parse JSON from model response. Using raw text.")
                answer = response_text
                thought_process = []
                
        except OpenAIError as e:
            logger.error(f"OpenAI API error: {str(e)}")
            raise RuntimeError(f"Failed to generate response: {str(e)}")

        # Calculate confidence as average of document scores
        confidence = (
            sum(src["score"] for src in sources) / len(sources) if sources else 0.0
        )

        return RagResponse(answer=answer, confidence=confidence, sources=sources, thought_process=thought_process)

    except Exception as e:
        logger.error(f"RAG pipeline error: {str(e)}")
        raise


# Interactive CLI for user queries
def main():
    print("Welcome to the RAG Movie Review Assistant!")
    print("Ask questions about IMDB movie reviews (type 'quit' to exit).")

    while True:
        query = input("\nAsk a question about a movie review: ").strip()
        if query.lower() == "quit":
            print("Goodbye!")
            break
        if not query:
            print("Please enter a valid question.")
            continue

        try:
            result = run_rag(query)
            print(f"\nAnswer: {result.answer}")
            print(f"\nConfidence: {result.confidence:.2%}")
            print("\nThought process:")
            for step in result.thought_process:
                print(f"  - {step}")
            print("\nSources:")
            for i, source in enumerate(result.sources, 1):
                print(
                    f"  {i}. Sentiment: {source['sentiment']}, Score: {source['score']}"
                )
                print(f"     Text: {source['text'][:100]}...")  # Truncate for brevity
        except Exception as e:
            print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
