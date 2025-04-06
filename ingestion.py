from dotenv import load_dotenv
import pandas as pd
import re
import logging
import psutil
from tqdm import tqdm
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

load_dotenv()


# Function to check and log current memory usage
def log_memory_usage(stage):
    process = psutil.Process()
    mem_in_mb = process.memory_info().rss / (1024 * 1024)
    logging.info(f"[{stage}] Current memory usage: {mem_in_mb:.2f} MB")


# Function to clean the content of a text column
def clean_text(text):
    """
    Clean the text by removing URLs and unnecessary characters.
    Args:
        text (str): The text to clean
    Returns:
        str: The cleaned text
    """
    # Remove URLs and unnecessary characters
    return re.sub(r'\{"url":.*?\}', "", text)


def process_csv_to_vectorstore(csv_file: str, collection_name: str) -> Chroma:
    """
    Process a CSV file and create a vector store from its contents.

    Args:
        csv_file (str): Path to the CSV file
        collection_name (str): Name for the Chroma collection

    Returns:
        Chroma: The created vector store
    """
    docs = []
    logging.info("Starting to process CSV file...")

    try:
        # Read and clean CSV
        logging.info(f"Reading file: {csv_file}")
        df = pd.read_csv(csv_file, skip_blank_lines=True)
        df = df.dropna(axis=1, how="all")
        df = df.dropna(axis=0, how="all")

        # Process each row: every row becomes a separate document
        for index, row in tqdm(df.iterrows(), desc="Processing rows", total=len(df)):
            row_text_columns = []
            for col in df.columns:
                cell = str(row[col])
                if col in ["data", "input"]:
                    cell = clean_text(cell)
                row_text_columns.append(f"{col}:\n{cell}")

            # Concatenate the content of all columns for the current row
            concatenated_text = "\n\n".join(row_text_columns)

            # Create a document with the processed content and add metadata (CSV file and row index)
            docs.append(
                Document(
                    page_content=concatenated_text,
                    metadata={"source": csv_file, "row": index},
                )
            )

        # Log memory usage after processing each file
        log_memory_usage(f"After processing {csv_file}")

        # Split documents
        logging.info("Splitting documents into chunks...")
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=250, chunk_overlap=0
        )
        # If don't split the documents the answer is more accurate
        # doc_splits = text_splitter.split_documents(docs)

        logging.info("Completed splitting documents.")
        log_memory_usage("After document splitting")

        # Store the embeddings in the vector store
        logging.info("Storing embeddings in the vector store...")
        vectorstore = Chroma.from_documents(
            documents=docs,
            collection_name=collection_name,
            embedding=OpenAIEmbeddings(),
            persist_directory="./.chroma",
        )
        logging.info("Vector store created successfully")
        log_memory_usage("After creating vector store")

        return vectorstore

    except Exception as e:
        logging.error(f"Error processing CSV file: {str(e)}")
        raise


def initialize_retriever(
    collection_name: str, persist_dir: str = "./.chroma"
) -> Chroma:
    """
    Initialize a Chroma retriever for vector store searches.

    Args:
        collection_name (str): Name of the Chroma collection
        persist_dir (str): Directory where the vector store is persisted

    Returns:
        Chroma: The initialized retriever
    """
    logging.info("Initializing retriever for vector store searches...")

    try:
        retriever = Chroma(
            collection_name=collection_name,
            persist_directory=persist_dir,
            embedding_function=OpenAIEmbeddings(),
        ).as_retriever()

        logging.info("Retriever initialized successfully.")
        log_memory_usage("After initializing retriever")

        return retriever

    except Exception as e:
        logging.error(f"Error initializing retriever: {str(e)}")
        raise


def setup_vectorstores(csv_config: dict) -> dict:
    """
    Set up multiple vector stores from CSV files.

    Args:
        csv_config (dict): Dictionary mapping CSV files to collection names

    Returns:
        dict: Dictionary of created vector stores
    """
    vectorstores = {}
    for csv_file, collection_name in csv_config.items():
        try:
            vectorstores[collection_name] = process_csv_to_vectorstore(
                csv_file=csv_file, collection_name=collection_name
            )
        except Exception as e:
            logging.error(f"Failed to process {csv_file}: {str(e)}")
    return vectorstores


def setup_retrievers(collections: list, persist_dir: str = "./.chroma") -> dict:
    """
    Initialize retrievers for multiple collections.

    Args:
        collections (list): List of collection names
        persist_dir (str): Directory where vector stores are persisted

    Returns:
        dict: Dictionary of initialized retrievers
    """
    retrievers = {}
    for collection in collections:
        try:
            retrievers[collection] = initialize_retriever(
                collection_name=collection, persist_dir=persist_dir
            )
        except Exception as e:
            logging.error(f"Failed to initialize retriever for {collection}: {str(e)}")
    return retrievers


def query_and_display_results(
    retriever: Chroma, query: str, collection_name: str = None
):
    """
    Execute a query and display results in a formatted way.

    Args:
        retriever (Chroma): The retriever to use
        query (str): Query string to search for
        collection_name (str, optional): Name of the collection being queried
    """
    logging.info(f"Executing query: {query}")
    results = retriever.get_relevant_documents(query)

    header = f"\n=== Query Results"
    if collection_name:
        header += f" from {collection_name}"
    header += " ==="

    print(header)
    print(f"Query: {query}")
    print(f"Found {len(results)} relevant documents\n")

    for i, doc in enumerate(results, 1):
        print(f"\nDocument {i}")
        print("=" * 50)
        print(f"Source: {doc.metadata['source']}")
        print(f"Row: {doc.metadata.get('row', 'N/A')}")
        print("\nContent:")
        print("-" * 50)
        print(doc.page_content)
        print("-" * 50)


def query_multiple_retrievers(retrievers: dict, query: str):
    """
    Execute the same query across multiple retrievers and display results.

    Args:
        retrievers (dict): Dictionary of retrievers with their collection names
        query (str): Query string to search for
    """
    print(f"\n{'='*20} Executing Query Across All Collections {'='*20}")
    print(f"Query: {query}\n")

    for collection_name, retriever in retrievers.items():
        print(f"\n{'='*20} Collection: {collection_name} {'='*20}")
        query_and_display_results(
            retriever=retriever, query=query, collection_name=collection_name
        )


if __name__ == "__main__":
    # Configuration
    csv_config = {
        "./assets/file_classifier_entries_upd.csv": "rag-advanced-file1",
        "./assets/file_classifier_insights.csv": "rag-advanced-file2",
        "./assets/file_classifier.csv": "rag-advanced-file3",
    }

    # Setup vector stores, comment this code once the vector stores are created
    vectorstores = setup_vectorstores(csv_config)
    logging.info("Completed processing CSV files.")
    log_memory_usage("After CSV processing")

    # Setup retrievers
    retrievers = setup_retrievers(list(csv_config.values()))

    # Example query across all retrievers
    test_query = "Give me some evaluated inputs where the model got it wrong"
    query_multiple_retrievers(retrievers=retrievers, query=test_query)
