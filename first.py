from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_ollama import ChatOllama

model = ChatOllama(model="llama3.2")

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a facts expert who know so much about {animal}"),
        ("human", "Tell me {fact_count} facts"),
    ]
)

chain = prompt_template | model | StrOutputParser()
result = chain.invoke({"animal": "tiger", "fact_count": 3})
print(result)