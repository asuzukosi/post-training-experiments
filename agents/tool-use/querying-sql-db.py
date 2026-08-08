import os
from operator import itemgetter
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai.chat_models import ChatOpenAI
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser



template = """
Based on the table schema below, write a SQL query that would answer the user's question:
{schema}

Question: {question}
SQL Query:
"""

prompt  = ChatPromptTemplate.from_template(template)

db = SQLDatabase.from_uri("sqlite://///Chinook.db")

def get_schema(_):
    return db.get_table_info()

def run_query(query):
    return db.run(query)

model = ChatOpenAI()

sql_response = (
    RunnablePassthrough.assign(schema=get_schema)
    | prompt
    | model.bind(stop=["\nSQLResult:"])
    | StrOutputParser()
)

template = """
Based on the table schema below, question, sql query, and sql response, write a natural language response:
{schema}

Question: {question}
SQL Query: {query}
SQL Resposne: {response}
"""
prompt_response = ChatPromptTemplate.from_template(template)


full_chain = (
    RunnablePassthrough.assign(query=sql_response).assign(
        schema=get_schema,
        response=lambda x: db.run(x["query"])
    )
    | prompt_response
    | model | StrOutputParser()
)

print(full_chain.invoke({"question": "How many employees are there?"}))

