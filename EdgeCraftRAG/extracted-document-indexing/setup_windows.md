# Windows Setup Guide for Document Indexing

This guide provides detailed instructions for setting up the document parsing and indexing system on Windows.

## Prerequisites

- **Windows 11** (64-bit)
- **Python 3.12**
- **pip** (Python package installer)

## Step-by-Step Installation

### 1. Install Python

1. Download Python from [python.org](https://www.python.org/downloads/)
2. **Important:** Check "Add Python to PATH" during installation
3. Verify installation:
   ```powershell
   python --version
   pip --version
   ```

### 2. Install System Dependencies (Required for Full Functionality)

#### A. Install Tesseract OCR (for image text extraction)

Tesseract is required for extracting text from images in PDFs and scanned documents.

**Installation:**
1. Download the latest installer from [UB-Mannheim Tesseract](https://github.com/UB-Mannheim/tesseract/wiki)
   - Direct link: https://github.com/UB-Mannheim/tesseract/wiki
   - Download: `tesseract-ocr-w64-setup-v5.x.x.exe` (64-bit)

2. Run the installer
   - Default installation path: `C:\Program Files\Tesseract-OCR`

3. Add to PATH:
   ```powershell
   # Open PowerShell as Administrator and run:
   [Environment]::SetEnvironmentVariable("Path", $env:Path + ";C:\Program Files\Tesseract-OCR", "Machine")
   ```

4. Verify installation:
   ```powershell
   # Restart PowerShell, then run:
   tesseract --version
   ```

5. Set Tesseract path in Python (if needed):
   ```python
   import pytesseract
   pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
   ```

#### B. Install Poppler (for PDF processing)

Poppler is required for converting PDF pages to images.

**Installation:**
1. Download Poppler for Windows from [oschwartz10612/poppler-windows](https://github.com/oschwartz10612/poppler-windows/releases/)
   - Download: `Release-xx.xx.x-x.zip`

2. Extract to a permanent location:
   ```
   C:\Program Files\poppler
   ```

3. Add to PATH:
   ```powershell
   # Open PowerShell as Administrator:
   [Environment]::SetEnvironmentVariable("Path", $env:Path + ";C:\Program Files\poppler\Library\bin", "Machine")
   ```

4. Verify installation:
   ```powershell
   # Restart PowerShell, then run:
   pdftoppm -v
   ```

#### C. Install LibreOffice (Optional - for better DOC/PPT conversion)

LibreOffice improves parsing quality for DOC, PPT, and other Office formats.

**Installation:**
1. Download from [LibreOffice website](https://www.libreoffice.org/download/download/)
2. Run the installer (default settings are fine)
3. Installation path is typically: `C:\Program Files\LibreOffice`

The `unstructured` library will automatically detect and use LibreOffice if available.

### 3. Create Virtual Environment (Recommended)

```powershell
# Navigate to project directory
cd C:\Users\intel\Desktop\GenAIExamples\EdgeCraftRAG\extracted-document-indexing

# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\Activate.ps1

# If you get execution policy error, run:
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

### 4. Install Python Dependencies

```powershell
# Make sure virtual environment is activated
pip install --upgrade pip

# Install all dependencies
pip install -r requirements.txt

# This will take a few minutes as it downloads all packages
```

**Note:** If you encounter errors with `unstructured[all-docs]`, try installing separately:
```powershell
pip install unstructured
pip install "unstructured[pdf]"
pip install "unstructured[docx]"
pip install "unstructured[pptx]"
```

### 5. Verify Installation

Create a test file `test_setup.py`:

```python
import sys
print(f"Python version: {sys.version}")

# Test imports
try:
    import llama_index
    print("✓ llama-index installed")
except ImportError as e:
    print(f"✗ llama-index error: {e}")

try:
    import unstructured
    print("✓ unstructured installed")
except ImportError as e:
    print(f"✗ unstructured error: {e}")

try:
    import faiss
    print("✓ faiss installed")
except ImportError as e:
    print(f"✗ faiss error: {e}")

try:
    import pytesseract
    pytesseract.get_tesseract_version()
    print("✓ Tesseract OCR accessible")
except Exception as e:
    print(f"✗ Tesseract error: {e}")
    print("  Make sure Tesseract is installed and in PATH")

try:
    from pdf2image import convert_from_path
    print("✓ Poppler accessible")
except Exception as e:
    print(f"✗ Poppler error: {e}")
    print("  Make sure Poppler is installed and in PATH")

print("\n✅ Setup verification complete!")
```

Run the test:
```powershell
python test_setup.py
```

## Common Issues and Solutions

### Issue 1: "Tesseract not found"
**Solution:** Add Tesseract to PATH or set it explicitly in code:
```python
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
```

### Issue 2: "Unable to get page count. Is poppler installed?"
**Solution:** Add Poppler's bin directory to PATH:
```powershell
$env:Path += ";C:\Program Files\poppler\Library\bin"
```

### Issue 3: ImportError with unstructured
**Solution:** Some Windows systems need explicit installation:
```powershell
pip install --upgrade unstructured
pip install python-magic-bin  # Windows-specific
```

### Issue 4: "Microsoft Visual C++ 14.0 is required"
**Solution:** Install Microsoft C++ Build Tools:
- Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/
- Select "Desktop development with C++" workload

### Issue 5: FAISS installation fails
**Solution:** Use the CPU-only version:
```powershell
pip install faiss-cpu
```

## Default: Using Local OpenVINO Models (No API Key Required)

The project uses Intel-optimized OpenVINO embeddings by default:

```powershell
# Install additional packages
pip install sentence-transformers
```

Modify your code to use HuggingFace embeddings:
```python
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Use local embedding model (no API key needed)
embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)
```

## Performance Tips for Windows

1. **Use SSD storage** for better file I/O performance
2. **Disable Windows Defender** scanning for your working directory (temporarily)
3. **Use "fast" strategy** for quick testing:
   ```python
   parser = DocumentParser(use_hi_res_strategy=False)
   ```
4. **Process files in batches** to avoid memory issues with large documents

## Next Steps

1. Create a `documents/` folder and add some test files
2. Run the example:
   ```powershell
   python example_usage.py
   ```
3. Check the generated `extracted_images/` and `faiss_index/` directories

## Getting Help

If you encounter issues:
1. Check that all system dependencies are in PATH
2. Verify Python package versions with `pip list`
3. Try running examples one at a time
4. Check the project README.md for additional information

---

**Last Updated:** January 2026  
**Tested On:** Windows 11, Python 3.12
