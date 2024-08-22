from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import Settings, StorageContext, VectorStoreIndex, SimpleDirectoryReader
from llama_index.vector_stores.lancedb import LanceDBVectorStore
from llama_index.core.node_parser import SentenceSplitter, SemanticSplitterNodeParser
from llama_index.core import KnowledgeGraphIndex
from llama_index.graph_stores.kuzu import KuzuGraphStore
from lancedb.rerankers import ColbertReranker

import lancedb
import kuzu


 # Ollama LLM
llm = Ollama(
    model="mistral-nemo",  # or any other model you have in Ollama
    temperature=0.0,
    request_timeout=120.0,
 )

# Ollama Embedding
embed_model = OllamaEmbedding(
    model_name="mxbai-embed-large",  # or any other model you have in Ollama
    request_timeout=120.0,
)

splitter = SemanticSplitterNodeParser(
    buffer_size=1, breakpoint_percentile_threshold=95, embed_model=embed_model
)

base_splitter = SentenceSplitter(chunk_size=1000, chunk_overlap=50)

Settings.llm = llm
Settings.embed_model = embed_model

documents = SimpleDirectoryReader(input_dir="./data", recursive=True).load_data()

vector_store = LanceDBVectorStore(
    uri="./.db/.lancedb",
    reranker=ColbertReranker(),
    mode="overwrite",
    query_type="hybrid"
)

# Create Kuzu graph store
kuzu_db = kuzu.Database("./.db/.kuzudb")
graph_store = KuzuGraphStore(database=kuzu_db)

# Create a storage context with both vector store and graph store
storage_context = StorageContext.from_defaults(
    vector_store=vector_store,
    graph_store=graph_store
)
# Create a knowledge graph index
kg_index = KnowledgeGraphIndex.from_documents(
    documents,
    storage_context=storage_context,
    include_embeddings=True,
    max_triplets_per_chunk=10,
    # kg_triplet_extract_fn=lambda x: x[:100],  # could limit triplets to avoid duplicates
)

# Create a hybrid query engine that combines vector and graph search
hybrid_query_engine = kg_index.as_query_engine(
    include_text=True,
    response_mode="tree_summarize",
    verbose=True
)

# Create a chat engine
chat_engine = hybrid_query_engine.as_chat_engine(chat_mode="context", verbose=True)

    # Example usage
while True:
    user_input = input("Your question (or 'quit' to exit): ")
    if user_input.lower() == 'quit':
        break
    response = chat_engine.chat(user_input)
    print("Assistant:", response.response)

if __name__ == "__main__":
    main()