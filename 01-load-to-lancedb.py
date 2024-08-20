# This is a functional Azure demo
# only works when llama_index is installed from pip3, not from conda-forge
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.core import Settings, StorageContext, VectorStoreIndex, SimpleDirectoryReader
from llama_index.vector_stores.lancedb import LanceDBVectorStore
from llama_index.core.node_parser import SentenceSplitter, SemanticSplitterNodeParser
from llama_index.core.ingestion import IngestionPipeline
from lancedb.rerankers import ColbertReranker
import streamlit as st
import lancedb
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
    uri="./lancedb/.lancedb",
    reranker=ColbertReranker(),
    mode="overwrite",
    query_type="hybrid"
)

storage_context = StorageContext.from_defaults(vector_store=vector_store)

pipeline = IngestionPipeline(
    transformations=[
        splitter,
        base_splitter
    ],
    vector_store=vector_store,
)

pipeline.run(documents=documents)

# Create a vector index from the documents (try later to use from_vector_store or from_table)
vector_index = VectorStoreIndex.from_documents(
    documents=documents,
    storage_context=storage_context,
)

# query_engine = vector_index.as_query_engine(query_mode="hybrid")
#
# response = query_engine.query("What are the components of a forensic evaluation?")
# print(response)
#


chat_engine = vector_index.as_chat_engine(chate_mode="context", llm=llm, verbose=True)

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