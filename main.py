from dotenv import load_dotenv
import os
import logging
import psutil

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

from graph.graph import app as graph_app

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


# Create a FastAPI instance
api = FastAPI(title="RAG API", version="1.0")

# Add CORS middleware
api.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Your frontend URL
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define request schema
class QuestionRequest(BaseModel):
    question: str


@api.post("/generate")
async def generate_response(request: QuestionRequest):
    try:
        logging.info("Running the app")
        log_memory_usage("Before invocation")

        # Invoke the graph application using the question from the API request
        result = graph_app.invoke(input={"question": request.question})
        logging.info(f"App result: {result}")

        # Extract the generated documents from the result
        generation = result.get("generation", "N/A")
        logging.info(f"Generated documents: {generation}")

        log_memory_usage("After invocation")
        return generation
    except Exception as e:
        logging.error("Error during generation: %s", e)
        raise HTTPException(status_code=500, detail="Error generating response")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(api, host="127.0.0.1", port=8000)
