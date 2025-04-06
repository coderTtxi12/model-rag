from typing import Any, Dict

from graph.chains.retrieval_grader import retrieval_grader
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


def grade_documents(state: GraphState) -> Dict[str, Any]:
    """
    Determines whether the retrieved documents are relevant to the question

    Args:
        state (dict): The current graph state

    Returns:
        Dict[str, Any]: Updated state with:
            - filtered_docs: list of relevant documents
            - results: True if relevant documents found, False otherwise
            - question: original question maintained
    """

    logging.info("Starting Corrective RAG process")
    log_memory_usage("Start grading")

    question = state["question"]
    documents = state["documents"]
    logging.info(f"Processing {len(documents)} documents for question: {question}")

    filtered_docs = []
    results = False

    for d in documents:
        score = retrieval_grader.invoke(
            {"question": question, "document": d.page_content}
        )
        grade = score.binary_score
        if grade.lower() == "yes":
            logging.info("------GRADE: DOCUMENT RELEVANT-----")
            filtered_docs.append(d)
        else:
            logging.info("-----GRADE: DOCUMENT NOT RELEVANT-----")
            continue

    # Set results to True if we found any relevant documents
    if filtered_docs:
        logging.info(f"Found {len(filtered_docs)} RELEVANT DOCUMENTS")
        results = True
    else:
        logging.warning("No relevant documents found")

    log_memory_usage("End grading")
    # The final return happens after the loop completes. It processes all documents first, then returns the dictionary
    return {"documents": filtered_docs, "question": question, "results": results}
