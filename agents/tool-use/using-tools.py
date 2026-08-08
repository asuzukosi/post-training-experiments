from langchain.tools import DuckDuckGoSearchRun
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv


search = DuckDuckGoSearchRun()

load_dotenv()
template = """
Turn the following user input into a search query for a search engine

{input}
"""

prompt = ChatPromptTemplate.from_template(template)
model = ChatOpenAI()


chain = prompt | model | StrOutputParser() | search

print(chain.invoke({"input": "I'd like to figure out what games are on tonight"}))