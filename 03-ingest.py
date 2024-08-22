# Separate the ingestion process from the querying process

from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import Settings, StorageContext, VectorStoreIndex, SimpleDirectoryReader
from llama_index.vector_stores.lancedb import LanceDBVectorStore
from llama_index.core.node_parser import SentenceSplitter, SemanticSplitterNodeParser
from llama_index.core import KnowledgeGraphIndex
from llama_index.graph_stores.kuzu import KuzuGraphStore
from lancedb.rerankers import ColbertReranker
from llama_index.core.extractors import TitleExtractor, KeywordExtractor

import lancedb
import kuzu
import os
import shutil
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Ollama LLM
llm = Ollama(
    model="mistral-nemo",
    temperature=0,
    request_timeout=120.0,
)

# Ollama Embedding
embed_model = OllamaEmbedding(
    model_name="mxbai-embed-large",
    request_timeout=120.0,
)

splitter = SemanticSplitterNodeParser(
    buffer_size=1, breakpoint_percentile_threshold=95, embed_model=embed_model
)

base_splitter = SentenceSplitter(chunk_size=1000, chunk_overlap=50)

Settings.llm = llm
Settings.embed_model = embed_model

def safe_extract_triplets(text):
    try:
        # Use a more robust triplet extraction method
        extractor = KeywordExtractor(
            keywords=["subject", "predicate", "object"],
            required_keywords=["subject", "object"]
        )
        extracted_info = extractor.extract(text)
        
        # Handle both list and dictionary outputs
        if isinstance(extracted_info, dict):
            extracted_triplets = [extracted_info]
        elif isinstance(extracted_info, list):
            extracted_triplets = extracted_info
        else:
            logger.exception(f"Unexpected extraction result type: {type(extracted_info)}")
            return []
        
        # Filter and format the extracted triplets
        triplets = []
        for triplet in extracted_triplets:
            if isinstance(triplet, dict):
                subject = triplet.get("subject", "")
                predicate = triplet.get("predicate", "is related to")
                object_ = triplet.get("object", "")
                if subject and object_:
                    triplets.append((subject, predicate, object_))
            else:
                logger.exception(f"Unexpected triplet format: {triplet}")
        
        logger.debug(f"Extracted {len(triplets)} triplets")
        return triplets[:100]  # Limit to 100 triplets
    except Exception as e:
        logger.exception(f"Error extracting triplets: {e}")
        return []

def ingest_documents():
    documents = SimpleDirectoryReader(input_dir="./data", recursive=True).load_data()

    vector_store = LanceDBVectorStore(
        uri="./.db/.lancedb",
        reranker=ColbertReranker(),
        mode="overwrite",
        query_type="hybrid"
    )

    # Clear existing Kuzu database
    kuzu_db_path = "./.db/.kuzudb"
    if os.path.exists(kuzu_db_path):
        shutil.rmtree(kuzu_db_path)

    # Create Kuzu graph store
    kuzu_db = kuzu.Database(kuzu_db_path)
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
        max_triplets_per_chunk=10,
        include_embeddings=True,
        kg_triplet_extract_fn=safe_extract_triplets,
        transformations=[TitleExtractor()],  # Add title extraction
    )

    # Save the index
    kg_index.storage_context.persist(persist_dir="./.db")

    print("Ingestion complete. Vector store and knowledge graph created and saved.")

if __name__ == "__main__":
    ingest_documents()