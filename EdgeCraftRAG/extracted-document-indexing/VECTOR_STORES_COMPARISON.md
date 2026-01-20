# Vector Store Comparison

## Quick Reference Matrix

| Feature | Default Vector ⭐ | FAISS ⭐ | Milvus ⭐ | ChromaDB |
|---------|------------------|----------|-----------|----------|
| **EdgeCraftRAG** | ✅ Built-in | ✅ Primary | ✅ Production | ❌ Not included |
| **Setup** | Instant | Easy | Medium | Easy |
| **Persistence** | ❌ Memory only | Manual | ✅ Auto | ✅ Auto |
| **Windows** | ✅ Native | ✅ Native | ⚠️ Docker | ✅ Native |
| **Scale** | < 10K | < 1M | > 1M | < 500K |
| **Speed** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **CRUD** | ✅ Yes | ❌ Limited | ✅ Full | ✅ Full |
| **Multi-Project Sharing** | ❌ No | ❌ No | ✅ **Yes** | ✅ Yes |
| **Cost** | Free | Free | Free/Paid | Free |

⭐ = EdgeCraftRAG Supported

---

## Multi-Project Sharing on Same SUT

### Sharing Capability Summary

| Vector Store | Shareable | Isolation Method | Best For |
|--------------|-----------|------------------|----------|
| **Default Vector** | ❌ No | N/A | Single project only |
| **FAISS** | ⚠️ Limited | Separate files | Not recommended |
| **Milvus** | ✅ **Excellent** | Collections | ✅ Multi-project SUT |
| **ChromaDB** | ✅ Good | Collections | Local multi-project |

**Recommended for Shared SUT:** **Milvus** (EdgeCraftRAG's production choice)

### Key Points:

**Milvus - Best for Sharing:**
```python
# Single server, multiple projects
vector_store = MilvusVectorStore(
    uri="http://localhost:19530",           # Shared server
    collection_name="project_a_docs",       # Unique per project
    dim=384
)
```

**Benefits:**
- ✅ One Milvus instance serves all projects
- ✅ Collection-based isolation (like database tables)
- ✅ 60-70% resource savings vs separate instances
- ✅ EdgeCraftRAG naming: `kb_name + pipeline_name`

**Example Multi-Project Setup:**
```
SUT with Single Milvus (localhost:19530)
├── project_a_chatbot_docs (1M vectors)
├── project_b_search_docs (500K vectors)
└── project_c_rag_knowledge (2M vectors)

Resource Savings: 3 projects, 1 server = 60-70% less resources
```

---

## Detailed Comparison

### 1. Default Vector ⭐
**EdgeCraftRAG's DEFAULT_VECTOR**

**Pros:**
- ✅ Zero configuration
- ✅ Instant setup (1 line of code)
- ✅ Perfect for prototyping
- ✅ Built into LlamaIndex

**Cons:**
- ❌ No persistence (lost on restart)
- ❌ Limited to small datasets
- ❌ Single machine only

**Best For:**
- Quick demos
- Testing/development
- < 10,000 documents

**Code:**
```python
index = VectorStoreIndex(nodes=nodes, embed_model=embed_model)
```

---

### 2. FAISS ⭐
**EdgeCraftRAG's FAISS_VECTOR (Primary)**

**Pros:**
- ✅ Fast similarity search
- ✅ No server setup needed
- ✅ Works offline
- ✅ Low resource usage
- ✅ Native Windows support

**Cons:**
- ⚠️ Manual persistence
- ❌ Limited CRUD operations
- ❌ Single machine only

**Best For:**
- Local deployments
- Medium datasets (10K-1M docs)
- Development/testing

**Code:**
```python
faiss_index = faiss.IndexFlatL2(384)
vector_store = FaissVectorStore(faiss_index=faiss_index)
```

---

### 3. Milvus ⭐
**EdgeCraftRAG's MILVUS_VECTOR (Production)**

**Pros:**
- ✅ Massive scalability (billions of vectors)
- ✅ Distributed architecture
- ✅ Full CRUD operations
- ✅ Production-ready
- ✅ GPU acceleration support

**Cons:**
- ❌ Requires Docker/server setup
- ❌ Higher resource usage
- ⚠️ More complex configuration

**Best For:**
- Production systems
- Large scale (> 1M docs)
- Enterprise applications

**Code:**
```python
vector_store = MilvusVectorStore(
    uri="http://localhost:19530",
    collection_name="documents",
    dim=384
)
```

---

### 4. ChromaDB
**Additional Option (Not in EdgeCraftRAG)**

**Pros:**
- ✅ Automatic persistence
- ✅ Easy to use
- ✅ Built-in metadata filtering
- ✅ Native Windows support
- ✅ Good for local projects

**Cons:**
- ⚠️ Not in EdgeCraftRAG
- ❌ Limited scalability vs Milvus
- ⚠️ Slower than FAISS

**Best For:**
- Local projects needing persistence
- Medium datasets (10K-500K docs)
- Solo developers

**Code:**
```python
client = chromadb.PersistentClient(path="./chroma_db")
vector_store = ChromaVectorStore(chroma_collection=collection)
```

---

## Decision Tree

```
How many documents?
│
├─ < 10K documents
│   └─► Use: Default Vector (fastest)
│
├─ 10K - 100K documents
│   └─► Use: FAISS (EdgeCraftRAG default)
│
├─ 100K - 500K documents
│   ├─ Need persistence?
│   │   ├─ Yes → ChromaDB
│   │   └─ No → FAISS
│
└─ > 500K documents
    └─► Use: Milvus (EdgeCraftRAG production)
```

---

## EdgeCraftRAG Alignment

| Vector Store | EdgeCraftRAG | This Project | Status |
|--------------|--------------|--------------|---------|
| Default Vector | ✅ DEFAULT_VECTOR | ✅ Supported | ✅ Aligned |
| FAISS | ✅ FAISS_VECTOR | ✅ Default | ✅ Aligned |
| Milvus | ✅ MILVUS_VECTOR | ✅ Supported | ✅ Aligned |
| ChromaDB | ❌ Not included | ✅ Bonus option | ➕ Extra |

---

## Performance Comparison

### Indexing Speed (10K documents, 384d vectors)

| Store | Time | Memory |
|-------|------|--------|
| Default | ~10s | 50 MB |
| FAISS | ~12s | 60 MB |
| ChromaDB | ~15s | 80 MB |
| Milvus | ~20s | 100 MB |

### Query Speed (avg. per query)

| Store | Latency |
|-------|---------|
| Default | < 1ms |
| FAISS | < 2ms |
| ChromaDB | 5-10ms |
| Milvus | 10-20ms |

*Note: Milvus shines at scale (> 100K docs)*

---

## Windows Compatibility

| Store | Native | Docker | Notes |
|-------|--------|--------|-------|
| Default | ✅ Yes | N/A | Pure Python |
| FAISS | ✅ Yes | N/A | pip install faiss-cpu |
| ChromaDB | ✅ Yes | N/A | pip install chromadb |
| Milvus | ❌ No | ✅ Yes | Requires Docker Desktop |

---

## Recommendation Summary

**For Development:**
- Use **Default Vector** for quick tests
- Use **FAISS** for realistic testing

**For Production:**
- Small scale (< 100K): **FAISS**
- Medium scale (100K-1M): **FAISS** or **ChromaDB**
- Large scale (> 1M): **Milvus**

**For Multi-Project SUT:**
- Use **Milvus** with collection-based isolation
- 60-70% resource savings vs separate instances
- EdgeCraftRAG's production pattern

**EdgeCraftRAG Way:**
- Development: **FAISS**
- Production: **Milvus**
- Prototyping: **Default Vector**

---

## Code Size Comparison

```python
# Default Vector: 1 line
index = VectorStoreIndex(nodes, embed_model)

# FAISS: 4 lines
faiss_index = faiss.IndexFlatL2(384)
vector_store = FaissVectorStore(faiss_index)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex(nodes, storage_context, embed_model)

# Milvus: 4 lines
vector_store = MilvusVectorStore(uri="...", dim=384, collection_name="docs")
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex(nodes, storage_context, embed_model)

# ChromaDB: 5 lines
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.create_collection("documents")
vector_store = ChromaVectorStore(chroma_collection=collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex(nodes, storage_context, embed_model)
```

---

**Generated:** January 2026  
**Version:** 1.1  
**Project:** EdgeCraftRAG Document Indexing
