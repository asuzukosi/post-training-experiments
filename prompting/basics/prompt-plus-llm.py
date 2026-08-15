"""
Simplest composition is just combining a prompt and models to create a chain that takes user
input, adds it to a prompt, passes it to a model and returns the returns the raw model output
"""
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain.output_parsers.openai_functions import JsonKeyOutputFunctionsParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

from dotenv import load_dotenv


load_dotenv()



prompt = ChatPromptTemplate.from_template("Tell me a joke about {foo}")
model = ChatOpenAI()

functions = [
    {
        "name": "joke",
        "description": "A joke",
        "parameters": {
            "type": "object",
            "properties": {
                "setup": {"type": "string", "description": "The setup for the joke"},
                "punchline": {
                    "type": "string",
                    "description": "The punchline for the joke",
                },
            },
            "required": ["setup", "punchline"],
        },
    }
]


map_ = RunnableParallel(foo=RunnablePassthrough())
chain = map_ | prompt | model.bind(function_call={"name":  "joke"}, functions=functions) | JsonKeyOutputFunctionsParser(key_name="punchline")
# chain = prompt | model | StrOutputParser()


print(chain.invoke({"foo": "bears"}, config={}))

