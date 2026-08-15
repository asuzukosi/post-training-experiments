import os
from operator import itemgetter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, RunnableParallel, Runnable
from langchain_core.embeddings import Embeddings
from typing import List
from langchain_core.runnables.config import run_in_executor
from langchain.schema import format_document
from langchain_core.messages import AIMessage, HumanMessage, get_buffer_string
from langchain.prompts.prompt import PromptTemplate
from langchain.memory import ConversationBufferMemory



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


# build the vector store
vectorstore = FAISS.from_texts(
    ["harrison worked at kensho"], embedding=CustomEmbeddingClass()
)

# build retreiver from the vector store
retreiver = vectorstore.as_retriever()

memory = ConversationBufferMemory(
    return_messages=True, output_key="answer", input_key="question"
)

loaded_memory = RunnablePassthrough.assign(
    chat_history=RunnableLambda(memory.load_memory_variables) | itemgetter("history")
)



template = """
Answer the question based only on the following context:
{context}

Question: {question}

"""


_template = """
Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:
"""

CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)
ANSWER_PROMPT = ChatPromptTemplate.from_template(template)
DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(template="{page_content}")


def _combine_documents(
    docs, document_prompt=DEFAULT_DOCUMENT_PROMPT, document_separator="\n\n"
):
    doc_strings = [format_document(doc, document_prompt) for doc in docs]
    return document_separator.join(doc_strings)



standalone_question = {
    "standalone_question": {
        "question": lambda x:x["question"],
        "chat_history": lambda x: get_buffer_string(x["chat_history"])
    }
    | CONDENSE_QUESTION_PROMPT
    | ChatOpenAI(temperature=0)
    | StrOutputParser(),
}

retreived_documents = {
    "docs": itemgetter("standalone_question") | retreiver,
    "question": lambda x:x["standalone_question"]
}

final_inputs = {
    "context": lambda x: _combine_documents(x["docs"]),
    "question": itemgetter("question")
}

answer = {
    "answer": final_inputs | ANSWER_PROMPT | ChatOpenAI(),
    "docs": itemgetter("docs")
}

final_chain = loaded_memory | standalone_question | retreived_documents | answer


inputs = {"question": "Where did harrison work?"}
result = final_chain.invoke(inputs)

print(result)

memory.save_context(inputs, {"answer": result["answer"].content})

memory.load_memory_variables({})

inputs  = {"question": "but where did he really work?"}
result = final_chain.invoke(inputs)

print(result)


# _inputs = RunnableParallel(
#     standalone_question=RunnablePassthrough.assign(
#         chat_history=lambda x: get_buffer_string(x["chat_history"])
#     )
#     | CONDENSE_QUESTION_PROMPT
#     | ChatOpenAI(temperature=0)
#     | StrOutputParser()
# )

# _context = {
#     "context": itemgetter("standalone_question") | retreiver | _combine_documents, "question": lambda x: x["standalone_question"]
# }

# conversational_qa_chain = _inputs | _context | ANSWER_PROMPT | ChatOpenAI()


# conversational_qa_chain.invoke({
#     "question": "where did harrison work?", 
#     "chat_history": [],
# })

"""

AIMessage(content='Harrison was employed at Kensho.')

conversational_qa_chain.invoke(
    {
        "question": "where did he work?",
        "chat_history": [
            HumanMessage(content="Who wrote this notebook?"),
            AIMessage(content="Harrison"),
        ],
    }
)

AIMessage(content='Harrison worked at Kensho.')
    
"""
# prompt = ChatPromptTemplate.from_template(template)
# model = ChatOpenAI()

# chain  = ({"context": itemgetter("question") | retreiver, "question": itemgetter("question"), "language": itemgetter("language")}
#           | prompt
#           | model
#           | StrOutputParser())

# print(chain.invoke({"question": "where did harrison work", "language": "french"}))