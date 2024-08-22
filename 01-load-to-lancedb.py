# This is a functional Azure demo
# only works when llama_index is installed from pip3, not from conda-forge
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.core import Settings, StorageContext, VectorStoreIndex, SimpleDirectoryReader
from llama_index.vector_stores.lancedb import LanceDBVectorStore
from llama_index.core.node_parser import SentenceSplitter, SemanticSplitterNodeParser
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core import KnowledgeGraphIndex
from llama_index.graph_stores.kuzu import KuzuGraphStore
from lancedb.rerankers import ColbertReranker

import lancedb
import kuzu

import streamlit as st
import dotenv
import os

from datetime import datetime

st.header("ForensicRAG")

if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {"role": "system", "content": "You are a helpful assistant that answers questions about psychiatric forensic "
                                      "evaluations."},
        {"role": "assistant", "content": "How can I assist you today?"}
    ]

dotenv.load_dotenv()

llm = AzureOpenAI(
    model="gpt-4o",
    deployment_name="gpt-4o",
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version="2023-03-15-preview",
)

# You need to deploy your own embedding model as well as your own chat completion model
embed_model = AzureOpenAIEmbedding(
    model="text-embedding-3-large",
    deployment_name="text-embedding-3-small",
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version="2023-05-15",
    termperatre=0.0,
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
    max_triplets_per_chunk=10
)

# Create a hybrid query engine that combines vector and graph search
hybrid_query_engine = kg_index.as_query_engine(
    include_text=True,
    response_mode="tree_summarize",
    verbose=True
)

# Update the chat engine to use the hybrid query engine
chat_engine = hybrid_query_engine.as_chat_engine(chat_mode="context", verbose=True)

if prompt := st.chat_input("Your question"): # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages: # Display the prior chat messages
    with st.chat_message(message["role"]):
        st.write(message["content"])

# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = chat_engine.chat(prompt)
            st.write(response.response)
            message = {"role": "assistant", "content": response.response}
            st.session_state.messages.append(message) # Add response to message history