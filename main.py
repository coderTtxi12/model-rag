from dotenv import load_dotenv
import os
import logging
import psutil

load_dotenv()
from pprint import pprint

from graph.graph import app

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


question = "What are the main issues with the model?"


if __name__ == "__main__":

    # initial_state = {
    #     "question": question1,
    #     "documents": None,
    #     "web_search": False,
    #     "generation": "",
    # }

    # print(app.invoke(input=initial_state))
    logging.info("Running the app")
    result = app.invoke(input={"question": question})
    logging.info(f"App result: {result}")

    # Log the generated documents from the result
    generation = result.get("generation", "N/A")
    logging.info(f"Generated documents: {generation}")
