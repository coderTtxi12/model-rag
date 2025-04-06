from typing import Any, Dict
from graph.state import GraphState
from ingestion import query_all_retrievers
import logging
import psutil

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def log_memory_usage(stage: str) -> None:
    """Log current memory usage."""
    process = psutil.Process()
    mem_info = process.memory_info()
    mem_mb = mem_info.rss / (1024 * 1024)  # Convert to MB
    logging.info(f"[{stage}] Memory usage: {mem_mb:.2f} MB")


def retrieve(state: GraphState) -> Dict[str, Any]:
    logging.info("Starting document retrieval process")
    log_memory_usage("Start retrieval")

    question = state["question"]
    logging.info(f"Processing question: {question}")

    # Doing semantic search and returning the documents
    documents = query_all_retrievers(query=question)
    logging.info(f"Retrieved {len(documents)} documents")

    log_memory_usage("After retrieval")

    # Log a summary of retrieved documents
    for i, doc in enumerate(documents, 1):
        logging.debug(
            f"Document {i} from collection: {doc.metadata.get('collection', 'unknown')}"
        )
    # Updating the field of documents in the state
    # and adding the current question
    return {"documents": documents, "question": question}
