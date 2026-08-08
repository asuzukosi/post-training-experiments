from operator import itemgetter
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI 
from dotenv import load_dotenv

load_dotenv()


model = ChatOpenAI()
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful chatbot"),
    MessagesPlaceholder(variable_name="history"),
    ("human","{input}")
])

memory = ConversationBufferMemory(return_messages=True)
memory.load_memory_variables({})

chain = RunnablePassthrough.assign(
    history=RunnableLambda(memory.load_memory_variables) | itemgetter("history")
) | prompt | model

inputs = {"input": "hi im bob"}
response = chain.invoke(inputs)
print(response)

memory.save_context(inputs, {"output": response.content})

memory.load_memory_variables({})

inputs = {"input": "whats my name"}
response = chain.invoke(inputs)
print(response)