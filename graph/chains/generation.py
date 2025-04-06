from langchain import hub
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="o3-mini",
)

# Create a prompt template runnable from the raw prompt string
prompt_template = ChatPromptTemplate.from_template(
    """You are an assistant for question-answering. Only use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. Add lots of emojis in your answer so everything is friendly readable.
Question: {question} 
Context: {context} 
Answer:"""
)

# Pipe the result into StrOutputParser
generation_chain = prompt_template | llm | StrOutputParser()
