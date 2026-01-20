# Document Parser & Vector Indexing

A standalone, Windows-compatible document parsing and vector indexing system extracted from EdgeCraftRAG's UnstructedNodeParser. Parse TXT, PDF, DOCX, PPT files and index them into vector databases for RAG applications.

## Features

✅ **Multi-format Support**: TXT, PDF, DOCX, DOC, PPTX, PPT, XLSX, XLS, HTML, HTM, XML, Markdown, RST  
✅ **Advanced PDF Processing**: High-resolution parsing with OCR  
✅ **Image Extraction**: Extract and OCR images from PDFs and DOCX files  
✅ **Multi-language OCR**: English, Chinese Simplified, Chinese Traditional  
✅ **Configurable Chunking**: Customizable chunk size and overlap  
✅ **Vector Database Support**: Default Vector, FAISS, Milvus (EdgeCraftRAG-aligned) + ChromaDB  
✅ **Windows Compatible**: Native Windows support with detailed setup guide  
✅ **Standalone**: No EdgeCraftRAG dependencies required  

## Quick Start

### Installation

```powershell
# Clone or navigate to the project
cd EdgeCraftRAG/extracted-document-indexing

# Create virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

**⚠️ Important:** For full functionality (PDF OCR), you need to install system dependencies:
- **Tesseract OCR** - for image text extraction
- **Poppler** - for PDF processing

See [setup_windows.md](setup_windows.md) for detailed Windows installation instructions.

### Basic Usage

```python
from document_parser import DocumentParser

# Initialize parser
parser = DocumentParser(
    chunk_size=512,
    chunk_overlap=50,
    extract_images=True
)

# Parse a single file
nodes = parser.parse_file("document.pdf")
print(f"Extracted {len(nodes)} chunks")

# Parse entire directory
nodes = parser.parse_directory("./documents", recursive=True)

# Get statistics
stats = parser.get_stats(nodes)
print(stats)
```

### Index to Vector Database

```python
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.faiss import FaissVectorStore
import faiss

# Create FAISS index
dimension = 384  # OpenVINO embedding dimension (BAAI/bge-small-en-v1.5)
faiss_index = faiss.IndexFlatL2(dimension)
vector_store = FaissVectorStore(faiss_index=faiss_index)

# Index documents
index = VectorStoreIndex(nodes=nodes, vector_store=vector_store)

# Query
query_engine = index.as_query_engine()
response = query_engine.query("What is this document about?")
print(response)
```

## Project Structure

```
extracted-document-indexing/
├── document_parser.py       # Main parser class
├── utils.py                 # Helper functions (image extraction, etc.)
├── example_usage.py         # Complete examples
├── requirements.txt         # Python dependencies
├── setup_windows.md         # Detailed Windows setup guide
└── README.md               # This file
```

## Configuration Options

### DocumentParser Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `chunk_size` | int | 250 | Maximum characters per chunk |
| `chunk_overlap` | int | 50 | Character overlap between chunks |
| `extract_images` | bool | True | Extract images from PDFs |
| `image_output_dir` | str | "./extracted_images" | Directory for extracted images |
| `ocr_languages` | List[str] | ["eng", "chi_sim", "chi"] | OCR language codes |
| `use_hi_res_strategy` | bool | True | Use high-res parsing (slower but more accurate) |

### Example Configurations

**Fast Processing (no image extraction):**
```python
parser = DocumentParser(
    chunk_size=512,
    chunk_overlap=50,
    extract_images=False,
    use_hi_res_strategy=False
)
```

**High Accuracy (with OCR):**
```python
parser = DocumentParser(
    chunk_size=256,
    chunk_overlap=100,
    extract_images=True,
    use_hi_res_strategy=True,
    ocr_languages=["eng", "chi_sim", "fra"]  # Add French
)
```

**Large Documents:**
```python
parser = DocumentParser(
    chunk_size=1024,
    chunk_overlap=200,
    extract_images=False  # Skip images for speed
)
```

## Supported Vector Databases (Aligned with EdgeCraftRAG)

### 1. Default Vector (EdgeCraftRAG Built-in) ⭐
**Simplest option - same as EdgeCraftRAG's DEFAULT_VECTOR**

```python
from llama_index.core import VectorStoreIndex

# No vector store configuration needed
index = VectorStoreIndex(nodes=nodes, embed_model=embed_model)
```

**Use for:** Quick prototyping, small datasets, testing

### 2. FAISS (EdgeCraftRAG Primary) ⭐
**Local, fast - EdgeCraftRAG's FAISS_VECTOR**

```python
import faiss
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core import StorageContext

faiss_index = faiss.IndexFlatL2(384)  # Match embedding dimension
vector_store = FaissVectorStore(faiss_index=faiss_index)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex(nodes=nodes, storage_context=storage_context, embed_model=embed_model)
```

**Use for:** Local deployments, medium datasets (10K-1M vectors)

### 3. Milvus (EdgeCraftRAG Production) ⭐
**Scalable, distributed - EdgeCraftRAG's MILVUS_VECTOR**

```python
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.core import StorageContext

vector_store = MilvusVectorStore(
    uri="http://localhost:19530",
    collection_name="documents",
    dim=384
)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex(nodes=nodes, storage_context=storage_context, embed_model=embed_model)
```

**Use for:** Production systems, large scale (1M+ vectors), distributed deployments

### 4. ChromaDB (Additional Option)
**Local with persistence**

```python
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.create_collection("documents")
vector_store = ChromaVectorStore(chroma_collection=collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex(nodes=nodes, storage_context=storage_context, embed_model=embed_model)
```

**Use for:** Local projects with automatic persistence

⭐ = Same as EdgeCraftRAG

## Embedding Options (Aligned with EdgeCraftRAG)

### OpenVINO (Default - Intel-optimized, local)
**Same as EdgeCraftRAG's default embedding method**

```python
from llama_index.embeddings.huggingface_openvino import OpenVINOEmbedding

# Intel-optimized embeddings (recommended)
embed_model = OpenVINOEmbedding(
    model_name_or_path="BAAI/bge-small-en-v1.5",
    device="CPU"  # or "GPU" if available
)
```

**Benefits:**
- ✅ Intel CPU optimization
- ✅ No API key required
- ✅ Runs completely locally
- ✅ Same approach as EdgeCraftRAG
- ✅ 2-4x faster than standard HuggingFace on Intel CPUs

### HuggingFace Direct (Local - without OpenVINO)
```python
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
```

## Examples

Run the included examples:

```powershell
python example_usage.py
```

This demonstrates:
1. Parsing a single file
2. Parsing a directory of files
3. Indexing to FAISS vector database
4. Querying the indexed documents
5. Alternative vector store options

## Windows-Specific Notes

### System Requirements
- Windows 10/11 (64-bit)
- Python 3.9-3.11
- 4GB+ RAM recommended
- SSD for better performance

### Required System Dependencies

1. **Tesseract OCR** (for image text extraction)
   - Download: [UB-Mannheim Tesseract](https://github.com/UB-Mannheim/tesseract/wiki)
   - Add to PATH

2. **Poppler** (for PDF processing)
   - Download: [Poppler Windows](https://github.com/oschwartz10612/poppler-windows/releases/)
   - Add to PATH

3. **LibreOffice** (optional, for better DOC/PPT parsing)
   - Download: [LibreOffice](https://www.libreoffice.org/download/download/)

See [setup_windows.md](setup_windows.md) for detailed installation instructions.

## Performance Tips

1. **Use fast strategy for testing:**
   ```python
   parser = DocumentParser(use_hi_res_strategy=False)
   ```

2. **Disable image extraction for speed:**
   ```python
   parser = DocumentParser(extract_images=False)
   ```

3. **Process files in batches:**
   ```python
   for batch in chunks(file_list, 10):
       nodes = parser.parse_files(batch)
       # Process nodes
   ```

4. **Use appropriate chunk sizes:**
   - Small chunks (256-512): Better for precise retrieval
   - Large chunks (1024-2048): Better for context

## Troubleshooting

### Common Issues

**"Tesseract not found"**
```python
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
```

**"Unable to get page count. Is poppler installed?"**
```powershell
# Add Poppler to PATH
$env:Path += ";C:\Program Files\poppler\Library\bin"
```

**Out of memory errors**
- Reduce `chunk_size`
- Disable `extract_images`
- Process files one at a time
- Use `use_hi_res_strategy=False`

**Slow parsing**
- Set `use_hi_res_strategy=False`
- Set `extract_images=False`
- Use SSD storage
- Process smaller batches

## API Reference

### DocumentParser

#### Methods

**`parse_file(file_path: str) -> List[BaseNode]`**
Parse a single file and return chunks.

**`parse_files(file_paths: List[str], deduplicate: bool = True) -> List[BaseNode]`**
Parse multiple files.

**`parse_directory(directory_path: str, recursive: bool = True, file_patterns: Optional[List[str]] = None) -> List[BaseNode]`**
Parse all supported files in a directory.

**`parse_with_simple_chunker(file_path: str) -> List[BaseNode]`**
Alternative fast parsing method using sentence splitter.

**`get_stats(nodes: List[BaseNode]) -> Dict[str, Any]`**
Get statistics about parsed nodes.

## Dependencies

Core dependencies:
- `llama-index` - RAG framework
- `unstructured` - Document parsing
- `python-docx` - DOCX processing
- `pillow` - Image handling
- `faiss-cpu` - Vector database
- `pytesseract` - OCR wrapper
- `pdf2image` - PDF to image conversion

See [requirements.txt](requirements.txt) for complete list.

## Integration with Other Projects

This parser can be easily integrated into other projects:

```python
# In your project
from document_parser import DocumentParser

# Use in your RAG pipeline
parser = DocumentParser(chunk_size=512)
nodes = parser.parse_directory("./data")

# Continue with your own indexing logic
your_vector_store.add(nodes)
```
