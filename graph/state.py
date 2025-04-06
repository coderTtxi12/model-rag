from typing import List, TypedDict


class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        documents: list of documents
        results: whether there are relevant documents to the question
    """

    question: str
    generation: str
    documents: List[str]
    results: bool
