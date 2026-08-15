from langchain.utils.math import cosine_similarity
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.embeddings import Embeddings
from typing import List
from langchain_core.runnables.config import run_in_executor
import os
from dotenv import load_dotenv

load_dotenv()


class CustomEmbeddingClass(Embeddings):
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed search docs."""
        from openai import OpenAI
        client = OpenAI(base_url=os.environ.get("OPENAI_API_BASE"), 
                        api_key=os.environ.get("OPENAI_API_KEY"))
        embeddings = []
        for txt in texts:
            response = client.embeddings.create(
                input=txt,
                model="text-embedding-ada-002"
            )
            embedding = response.data[0].embedding["data"][0]["embedding"]
            embeddings.append(embedding)
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """Embed query text."""
        from openai import OpenAI
        client = OpenAI(base_url=os.environ.get("OPENAI_API_BASE"), 
                        api_key=os.environ.get("OPENAI_API_KEY"))
        response = client.embeddings.create(
                input=text,
                model="text-embedding-ada-002"
            )
        embedding = response.data[0].embedding["data"][0]["embedding"]
        return embedding
        
    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """Asynchronous Embed search docs."""
        return await run_in_executor(None, self.embed_documents, texts)

    async def aembed_query(self, text: str) -> List[float]:
        """Asynchronous Embed query text."""
        return await run_in_executor(None, self.embed_query, text)




physics_template = """You are a very smart physics professor. \
You are great at answering questions about physics in a concise and easy to understand manner. \
When you don't know the answer to a question you admit that you don't know.

Here is a question:
{query}"""

math_template = """You are a very good mathematician. You are great at answering math questions. \
You are so good because you are able to break down hard problems into their component parts, \
answer the component parts, and then put them together to answer the broader question.

Here is a question:
{query}"""

embeddings = CustomEmbeddingClass()

prompt_templates = [physics_template, math_template]
prompt_embeddings = embeddings.embed_documents(prompt_templates)

def prompt_router(input):
    query_embedding = embeddings.embed_query(input["query"])
    similarity = cosine_similarity([query_embedding], prompt_embeddings)[0]
    most_similar = prompt_templates[similarity.argmax()]
    print("Using MATH" if most_similar == math_template else "Using PHYSICS")
    return PromptTemplate.from_template(most_similar)

chain = (
    {"query": RunnablePassthrough()}
    | RunnableLambda(prompt_router)
    | ChatOpenAI()
    | StrOutputParser()
)

print(chain.invoke("What's a path integral"))