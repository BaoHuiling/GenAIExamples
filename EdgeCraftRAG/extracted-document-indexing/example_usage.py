"""
Example usage of DocumentParser with vector database indexing
Demonstrates parsing documents and inserting into FAISS vector store
"""

import os
from pathlib import Path
from typing import List

from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.schema import BaseNode
from llama_index.embeddings.huggingface_openvino import OpenVINOEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore
import faiss

from document_parser import DocumentParser


def example_1_parse_single_file():
    """Example 1: Parse a single PDF file"""
    print("=" * 60)
    print("Example 1: Parse a single file")
    print("=" * 60)

    # Initialize parser
    parser = DocumentParser(
        chunk_size=512,
        chunk_overlap=50,
        extract_images=True,
        image_output_dir="./extracted_images",
    )

    # Parse a file (replace with your actual file path)
    file_path = "sample_document.pdf"

    if not os.path.exists(file_path):
        print(f"⚠️  File not found: {file_path}")
        print("Please update the file_path variable with an actual file.")
        return None

    try:
        nodes = parser.parse_file(file_path)
        print(f"\n✓ Successfully parsed {len(nodes)} chunks")

        # Print stats
        stats = parser.get_stats(nodes)
        print(f"\nStatistics:")
        print(f"  - Total nodes: {stats['total_nodes']}")
        print(f"  - Total characters: {stats['total_characters']}")
        print(f"  - Average chunk size: {stats['average_chunk_size']:.2f}")

        # Preview first chunk
        if nodes:
            print(f"\n📄 First chunk preview:")
            print(f"  {nodes[0].get_content()[:200]}...")

        return nodes
    except Exception as e:
        print(f"❌ Error: {e}")
        return None


def example_2_parse_directory():
    """Example 2: Parse all files in a directory"""
    print("\n" + "=" * 60)
    print("Example 2: Parse all files in a directory")
    print("=" * 60)

    parser = DocumentParser(
        chunk_size=512, chunk_overlap=50, extract_images=False  # Faster without image extraction
    )

    # Parse directory (replace with your actual directory)
    directory_path = "./documents"

    if not os.path.exists(directory_path):
        print(f"⚠️  Directory not found: {directory_path}")
        print("Please create a './documents' folder with some files or update the path.")
        return None

    try:
        nodes = parser.parse_directory(directory_path, recursive=True, file_patterns=["*.pdf", "*.docx", "*.txt"])

        print(f"\n✓ Successfully parsed {len(nodes)} chunks from directory")

        stats = parser.get_stats(nodes)
        print(f"\nStatistics:")
        print(f"  - Files processed: {stats['unique_files']}")
        print(f"  - Total chunks: {stats['total_nodes']}")
        print(f"  - Total characters: {stats['total_characters']}")

        return nodes
    except Exception as e:
        print(f"❌ Error: {e}")
        return None


def example_3_index_to_faiss(nodes: List[BaseNode]):
    """Example 3: Index parsed nodes into FAISS vector database"""
    print("\n" + "=" * 60)
    print("Example 3: Index to FAISS Vector Database")
    print("=" * 60)

    if not nodes:
        print("⚠️  No nodes to index. Run example 1 or 2 first.")
        return

    try:
        # Create embedding model - Using OpenVINO (Intel-optimized, same as EdgeCraftRAG)
        print("\n⏳ Initializing OpenVINO embedding model...")
        embed_model = OpenVINOEmbedding(
            model_name_or_path="BAAI/bge-small-en-v1.5",
            device="CPU",  # Use GPU if available
        )
        print("✓ Embedding model loaded")
        
        # Get embedding dimension from model
        dimension = 384  # BAAI/bge-small-en-v1.5 dimension
        
        # Create FAISS index
        faiss_index = faiss.IndexFlatL2(dimension)

        # Create vector store
        vector_store = FaissVectorStore(faiss_index=faiss_index)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        # Create index and add documents
        print(f"\n⏳ Indexing {len(nodes)} chunks...")
        index = VectorStoreIndex(nodes=nodes, storage_context=storage_context, embed_model=embed_model)

        print(f"✓ Successfully indexed {len(nodes)} chunks to FAISS")

        # Save index to disk
        index.storage_context.persist(persist_dir="./faiss_index")
        print("✓ Index saved to './faiss_index'")

        return index
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nNote: OpenVINO embedding model will be downloaded on first use.")
        print("This may take a few minutes depending on your internet connection.")
        return None


def example_4_query_index(index: VectorStoreIndex):
    """Example 4: Query the indexed documents"""
    print("\n" + "=" * 60)
    print("Example 4: Query the Vector Database")
    print("=" * 60)

    if not index:
        print("⚠️  No index available. Run example 3 first.")
        return

    try:
        # Create query engine
        query_engine = index.as_query_engine(similarity_top_k=3)

        # Example queries
        queries = [
            "What is the main topic of the documents?",
            "Summarize the key points",
        ]

        for query in queries:
            print(f"\n❓ Query: {query}")
            response = query_engine.query(query)
            print(f"💬 Response: {response}")

    except Exception as e:
        print(f"❌ Error: {e}")


def example_5_alternative_embeddings():
    """Example 5: Alternative embedding model options"""
    print("\n" + "=" * 60)
    print("Example 5: Alternative Embedding Models")
    print("=" * 60)

    print("""
Available embedding options (aligned with EdgeCraftRAG):

1. OpenVINO (Default - Intel-optimized, local, no API key)
   from llama_index.embeddings.huggingface_openvino import OpenVINOEmbedding
   
   embed_model = OpenVINOEmbedding(
       model_name_or_path="BAAI/bge-small-en-v1.5",  # 384 dimensions
       device="CPU"  # or "GPU"
   )
   
   Popular models:
   - BAAI/bge-small-en-v1.5 (384d) - Fast, good quality
   - BAAI/bge-base-en-v1.5 (768d) - Better quality, slower
   - sentence-transformers/all-MiniLM-L6-v2 (384d) - Fast
   
2. HuggingFace Direct (Local, without OpenVINO optimization)
   from llama_index.embeddings.huggingface import HuggingFaceEmbedding
   
   embed_model = HuggingFaceEmbedding(
       model_name="BAAI/bge-small-en-v1.5"
   )

Recommended: OpenVINO for best CPU performance (same as EdgeCraftRAG)
    """)


def example_6_alternative_vector_stores():
    """Example 6: Alternative vector database options"""
    print("\n" + "=" * 60)
    print("Example 6: Alternative Vector Stores")
    print("=" * 60)

    print("""
Available vector store options (aligned with EdgeCraftRAG):

1. Default Vector (EdgeCraftRAG built-in - Simplest option)
   from llama_index.core import VectorStoreIndex
   
   # No vector store needed, uses LlamaIndex's default in-memory store
   index = VectorStoreIndex(nodes=nodes, embed_model=embed_model)
   # Best for: Quick prototyping, small datasets
   
2. FAISS (EdgeCraftRAG primary - Local, fast, no server)
   import faiss
   from llama_index.vector_stores.faiss import FaissVectorStore
   from llama_index.core import StorageContext
   
   faiss_index = faiss.IndexFlatL2(384)  # dimension
   vector_store = FaissVectorStore(faiss_index=faiss_index)
   storage_context = StorageContext.from_defaults(vector_store=vector_store)
   index = VectorStoreIndex(nodes=nodes, storage_context=storage_context, embed_model=embed_model)
   # Best for: Local deployments, medium datasets
   
3. Milvus (EdgeCraftRAG production - Scalable, distributed)
   from llama_index.vector_stores.milvus import MilvusVectorStore
   from llama_index.core import StorageContext
   
   vector_store = MilvusVectorStore(
       uri="http://localhost:19530",
       collection_name="documents",
       dim=384
   )
   storage_context = StorageContext.from_defaults(vector_store=vector_store)
   index = VectorStoreIndex(nodes=nodes, storage_context=storage_context, embed_model=embed_model)
   # Best for: Production, large scale, distributed systems
   
4. ChromaDB (Additional option - Local with persistence)
   import chromadb
   from llama_index.vector_stores.chroma import ChromaVectorStore
   from llama_index.core import StorageContext
   
   client = chromadb.PersistentClient(path="./chroma_db")
   collection = client.create_collection("documents")
   vector_store = ChromaVectorStore(chroma_collection=collection)
   storage_context = StorageContext.from_defaults(vector_store=vector_store)
   index = VectorStoreIndex(nodes=nodes, storage_context=storage_context, embed_model=embed_model)
   # Best for: Local projects with persistence needs

EdgeCraftRAG uses: Default Vector, FAISS, and Milvus
    """)


def main():
    """Run all examples"""
    print("\n🚀 Document Indexing Examples")
    print("=" * 60)

    # Example 1: Parse single file
    nodes = example_1_parse_single_file()

    # Example 2: Parse directory (uncomment if you have a documents directory)
    # nodes = example_2_parse_directory()

    # Example 3: Index to FAISS (requires nodes from Example 1 or 2)
    if nodes:
        index = example_3_index_to_faiss(nodes)

        # Example 4: Query the index
        if index:
            example_4_query_index(index)

    # Example 5: Show alternative embedding models
    example_5_alternative_embeddings()
    
    # Example 6: Show alternative vector stores
    example_6_alternative_vector_stores()

    print("\n" + "=" * 60)
    print("✅ Examples complete!")
    print("=" * 60)


if __name__ == "__main__":
    print("""
📌 Using Intel-optimized OpenVINO embeddings (same as EdgeCraftRAG)
   - No API key required
   - Runs locally on CPU
   - First run will download model (~120MB)
   - Optimized for Intel hardware
    """)
    
    main()
