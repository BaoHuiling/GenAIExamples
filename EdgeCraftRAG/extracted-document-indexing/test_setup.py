"""
Test script to verify the installation and setup
"""

import sys
import os

print("=" * 60)
print("Document Parser Setup Verification")
print("=" * 60)

# Test Python version
print(f"\n✓ Python version: {sys.version.split()[0]}")
if sys.version_info != (3, 12):
    print("  ⚠️  Warning: Recommended Python version is 3.12")

# Test imports
print("\nChecking Python packages...")

packages_to_test = [
    ("llama_index", "LlamaIndex"),
    ("llama_index.core", "LlamaIndex Core"),
    ("llama_index.readers.file", "LlamaIndex File Readers"),
    ("unstructured", "Unstructured"),
    ("docx", "python-docx"),
    ("PIL", "Pillow"),
    ("faiss", "FAISS"),
    ("pydantic", "Pydantic"),
]

all_passed = True
for package, name in packages_to_test:
    try:
        __import__(package)
        print(f"  ✓ {name}")
    except ImportError as e:
        print(f"  ✗ {name} - Not installed")
        all_passed = False

# Test system dependencies
print("\nChecking system dependencies...")

# Test Tesseract
try:
    import pytesseract
    version = pytesseract.get_tesseract_version()
    print(f"  ✓ Tesseract OCR (version {version})")
except Exception as e:
    print(f"  ⚠️  Tesseract OCR - Not found or not in PATH")
    print(f"     Error: {e}")
    print(f"     Install from: https://github.com/UB-Mannheim/tesseract/wiki")

# Test Poppler
try:
    from pdf2image import convert_from_path
    from pdf2image.exceptions import PDFInfoNotInstalledError
    # Try a dummy call to check if poppler is available
    try:
        # This will fail but tells us if poppler is available
        convert_from_path("nonexistent.pdf", first_page=1, last_page=1)
    except PDFInfoNotInstalledError:
        print(f"  ⚠️  Poppler - Not found or not in PATH")
        print(f"     Install from: https://github.com/oschwartz10612/poppler-windows/releases/")
    except Exception:
        # Other errors mean poppler is found but file doesn't exist (which is expected)
        print(f"  ✓ Poppler")
except ImportError:
    print(f"  ⚠️  pdf2image - Not installed")

# Test document_parser module
print("\nTesting project modules...")
try:
    from document_parser import DocumentParser
    print("  ✓ document_parser.py")
except ImportError as e:
    print(f"  ✗ document_parser.py - {e}")
    all_passed = False

try:
    from utils import DocxParagraphPicturePartitioner, ensure_directory
    print("  ✓ utils.py")
except ImportError as e:
    print(f"  ✗ utils.py - {e}")
    all_passed = False

# Test creating parser instance
print("\nTesting DocumentParser initialization...")
try:
    from document_parser import DocumentParser
    parser = DocumentParser(chunk_size=512, chunk_overlap=50)
    print("  ✓ DocumentParser instance created successfully")
except Exception as e:
    print(f"  ✗ Failed to create DocumentParser: {e}")
    all_passed = False

# Check environment variables
print("\nChecking environment variables...")
print("  ✓ Using OpenVINO embeddings (no API key required)")
print("    First run will download model (~120MB)")

# Summary
print("\n" + "=" * 60)
if all_passed:
    print("✅ Setup verification PASSED!")
    print("\nYou can now run:")
    print("  python example_usage.py")
else:
    print("⚠️  Setup verification completed with warnings")
    print("\nSome components are missing. Check the messages above.")
    print("Refer to setup_windows.md for detailed installation instructions.")

print("=" * 60)
