from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver

# from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph

from graph.chains.answer_grader import answer_grader
from graph.chains.hallucination_grader import hallucination_grader
from graph.consts import GENERATE, GRADE_DOCUMENTS, RETRIEVE
from graph.nodes import generate, grade_documents, retrieve
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


load_dotenv()


def decide_to_generate(state):

    logging.info("Assessing graded documents")
    log_memory_usage("Document assessment")

    if state["results"]:
        logging.info("DECISION: Relevant documents found - proceeding to generation")
        return GENERATE
    else:
        logging.warning("DECISION: No relevant documents found - ending process")
        return END


def grade_generation_grounded_in_documents_and_question(state: GraphState) -> str:

    logging.info("Starting hallucination check")
    log_memory_usage("Start grading")

    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    score = hallucination_grader.invoke(
        {"documents": documents, "generation": generation}
    )

    if hallucination_grade := score.binary_score:

        logging.info("Grade generation vs Question")

        score = answer_grader.invoke({"question": question, "generation": generation})
        if answer_grade := score.binary_score:
            logging.info("ASSESSMENT: Generation addresses question successfully")
            log_memory_usage("End grading - successful")
            return "useful"
        else:
            logging.warning("ASSESSMENT: Generation does not address question")
            log_memory_usage("End grading - inadequate")
            return "not useful"
    else:
        logging.warning("ASSESSMENT: Generation contains hallucinations - retrying")
        log_memory_usage("End grading - hallucination detected")
        return "hallucination"


# Initialize the graph
workflow = StateGraph(GraphState)

# Nodes
workflow.add_node(RETRIEVE, retrieve)
workflow.add_node(GRADE_DOCUMENTS, grade_documents)
workflow.add_node(GENERATE, generate)

# Entry point
workflow.set_entry_point(RETRIEVE)

# Edges
workflow.add_edge(RETRIEVE, GRADE_DOCUMENTS)


workflow.add_conditional_edges(
    GRADE_DOCUMENTS,
    decide_to_generate,
    {
        GENERATE: GENERATE,
        END: END,
    },
)


workflow.add_conditional_edges(
    GENERATE,
    grade_generation_grounded_in_documents_and_question,
    {
        "not useful": GENERATE,
        "useful": END,
        "hallucination": GENERATE,
    },
)


app = workflow.compile()

# app.get_graph().draw_mermaid_png(output_file_path="graph.png")
