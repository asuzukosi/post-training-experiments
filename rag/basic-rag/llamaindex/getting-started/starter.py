import os
from llama_index import VectorStoreIndex
from llama_index.readers.file.base import SimpleDirectoryReader
from llama_index.storage.storage_context import StorageContext
from llama_index import load_index_from_storage


PERSIST_DIR = "./storage"

if not os.path.exists(PERSIST_DIR):
    documents = SimpleDirectoryReader("data").load_data()
    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist(persist_dir=PERSIST_DIR)
else:
    storage_context =StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context)

query_engine =  index.as_query_engine()

while True:
    query = input("Enter your question: ")
    response = query_engine.query(query)
    print("Response :")
    print(response.response)