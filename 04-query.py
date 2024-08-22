from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import Settings, StorageContext, load_index_from_storage
from llama_index.vector_stores.lancedb import LanceDBVectorStore
from llama_index.graph_stores.kuzu import KuzuGraphStore
from lancedb.rerankers import ColbertReranker

import kuzu

# Ollama LLM
llm = Ollama(
    model="mistral-nemo",
    temperature=0.0,
    request_timeout=120.0,
)

# Ollama Embedding
embed_model = OllamaEmbedding(
    model_name="mxbai-embed-large",
    request_timeout=120.0,
)

Settings.llm = llm
Settings.embed_model = embed_model

def load_and_query():
    # Load the storage context
    storage_context = StorageContext.from_defaults(
        persist_dir="./.db"
    )

    # Load the existing index
    kg_index = load_index_from_storage(storage_context)

    # Create a chat engine
    chat_engine = kg_index.as_chat_engine(
        chat_mode="context",
        verbose=True,
        similarity_top_k=3,
        include_text=True,
        response_mode="tree_summarize"
    )

    # Example usage
    while True:
        user_input = input("Your question (or 'quit' to exit): ")
        if user_input.lower() == 'quit':
            break
        response = chat_engine.chat(user_input)
        print("Assistant:", response.response)

if __name__ == "__main__":
    load_and_query()