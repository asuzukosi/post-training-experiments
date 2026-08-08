from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from langchain_experimental.utilities import PythonREPL
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv


load_dotenv()

template = """
Write some python code to solve the users's problem.

Reutrn only the python code in markdown format, e.g.:

```python
...
```
"""

prompt = ChatPromptTemplate.from_messages([("system", template), ("human", "{input}")])

model = ChatOpenAI()


def _sanitize_output(text: str):
    """
    Return the python output code quote from output response
    """
    print(text)
    _, after = text.split("```python")
    return after.split("```")[0]

chain = prompt | model | StrOutputParser() | _sanitize_output | PythonREPL().run


print(chain.invoke({"input": "What is 2 plus 2"}))



