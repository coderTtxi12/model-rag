# Model Insights using RAG

This project implements a Retrieval-Augmented Generation (RAG) pipeline using a graph-based architecture. It leverages advanced techniques like **Corrective RAG** and **Self RAG** to enhance answer accuracy and reliability. The system is composed of modular nodes—retrieval, grading, and generation—connected through conditional flows. These nodes collaboratively process user queries, assess response quality, and iteratively improve the final output.

## Architecture

- **Graph-Based Workflow**  
  The pipeline is orchestrated with a graph (using LangGraph), where each node represents a distinct task:
  - **Retrieval Node**: Gathers relevant documents based on the user question.
  - **Grading Nodes**:
    - _Document Grader_: Evaluates document relevance.
    - _Hallucination & Answer Grader_: Checks if the generation is grounded in documents and addresses the question.
  - **Generation Node**: Uses a LangChain generation chain (with prompt templates and OpenAI’s ChatOpenAI) to produce the final answer.
- **Self-Containment & Modular Design**  
  Each component (nodes and chains) is implemented as an independent module. Components are composed using a pipe operator (`|`) for chaining operations, and conditional edges control the overall flow.
- **API Gateway**  
  A FastAPI application is provided as the entry point. The `/generate` endpoint accepts a question and returns the generated response.

## Tools and Libraries

- **FastAPI & Uvicorn**  
  Provides an asynchronous API that exposes the RAG service.
- **LangChain & LangGraph**  
  Used to define and connect the different nodes and chains in the RAG pipeline.
- **OpenAI's ChatOpenAI**  
  Powers the generation and grading chains for response formation and validation.
- **Pydantic**  
  Validates input schemas and structured outputs.
- **psutil**  
  Tracks memory usage at critical points to support debugging and performance tuning.
- **Poetry**  
  Manages dependencies and virtual environment.

## Running the Application

1. **Install Dependencies**  
   Make sure you have [Poetry](https://python-poetry.org/) installed. From the project root, install dependencies:

   ```bash
   poetry install
   ```

2. **Environment Variables**  
   Create a `.env` file at the project root with the necessary configuration. For example, include your API keys and model settings:

   ```env
   OPENAI_API_KEY=your-api-key-here
   LANGCHAIN_PROJECT=your-langchain-name-project
   LANGCHAIN_API_KEY=your-langchain-api-key
   LANGCHAIN_TRACING_V2=true
   LANGSMITH_TRACING=true
   LANGSMITH_ENDPOINT=your-langsmith-endpoint
   LANGSMITH_API_KEY=your-langsmith-api-key
   LANGSMITH_PROJECT=your-langsmith-project
   ```

3. **Run the FastAPI App**  
   Launch the application using Uvicorn via Poetry:

   ```bash
   poetry run uvicorn main:api --host 127.0.0.1 --port 8000
   ```

   Replace `main:api` with the correct module and FastAPI instance name if different.

4. **Access the API**  
   You can test the API using a tool like `curl` or Postman. For example, to call the `/generate` endpoint:

   ```bash
   curl -X POST "http://127.0.0.1:8000/generate" \
   -H "Content-Type: application/json" \
   -d '{"question": "What are the main issues with the model?"}'
   ```

   The API will return the generated response in JSON format.

## Project Structure

```
model-rag/
├── assets/
├── graph/
├── const.py
├── graph.py
├── state.py
├── chains/
│   ├── tests/
│   │   └── test_chains.py
│   ├── answer_grader.py
│   ├── generation.py
│   ├── hallucination_grader.py
│   └── retrieval_grader.py
├── nodes/
│   ├── generate.py
│   ├── grade_documents.py
│   └── retrieve.py
├── .gitignore
├── ingestion.py
├── main.py
├── pyproject.toml
├── README.md

```

- **main.py**  
  The entry point that configures logging, initializes FastAPI, and defines the `/generate` endpoint.
- **graph/**  
  Contains the graph-based workflow implementation, including:
  - **nodes/**: Modules for retrieval, generation, and grading.
  - **chains/**: Definition of chains for generation, grading, and retrieval.
  - **state.py**: Defines the shared state used across nodes (State Machine).
- **README.md**  
  This file.

## Contributing

Feel free to contribute by submitting issues or pull requests. Contributions are welcome!

## License

This project is licensed under the MIT License.
