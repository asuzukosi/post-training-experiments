from langchain.chains import OpenAIModerationChain
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI 
from dotenv import load_dotenv

load_dotenv()


moderate = OpenAIModerationChain()
model = ChatOpenAI()

prompt = ChatPromptTemplate.from_messages([("system", "repeat after me: {input}")])

chain = prompt | model 
print(chain.invoke({"input": "you are stupid"}))

# moderation chain no longer working as openai api expired  
# moderated_chain = chain | moderate 

# moderated_chain.invoke({"input": "you are stupid"})

