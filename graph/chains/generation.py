from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(temperature=0)

prompt = hub.pull("rlm/rag-prompt")


# Pipe the result into StrOutputParser
generation_chain = prompt | llm | StrOutputParser()
