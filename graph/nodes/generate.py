from typing import Any, Dict

from graph.chains.generation import generation_chain
from graph.state import GraphState
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


def generate(state: GraphState) -> Dict[str, Any]:
    logging.info("Starting response generation")
    log_memory_usage("Start generation")

    question = state["question"]
    documents = state["documents"]

    logging.info(f"Generating response for question: {question}")
    logging.info(f"Using {len(documents)} documents for context")

    generation = generation_chain.invoke({"context": documents, "question": question})

    logging.info("Response generated successfully")    
    log_memory_usage("End generation")
    
    return {"documents": documents, "question": question, "generation": generation}
