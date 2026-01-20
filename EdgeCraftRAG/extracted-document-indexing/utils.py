"""
Utility functions for document parsing
Includes image extraction from DOCX files
"""

import io
import os
from pathlib import Path
from typing import Iterator

from PIL import Image as PILImage
from unstructured.documents.elements import ElementMetadata, Image
from unstructured.partition.docx import DocxPartitionerOptions

try:
    from docx.text.paragraph import Paragraph
except ImportError:
    print("Warning: python-docx not installed. DOCX image extraction will not work.")
    Paragraph = None


class DocxParagraphPicturePartitioner:
    """
    Custom partitioner to extract images from DOCX paragraphs.
    This preserves images that might be lost with standard parsing.
    """

    @classmethod
    def iter_elements(cls, paragraph: Paragraph, opts: DocxPartitionerOptions) -> Iterator[Image]:
        """
        Extract images from DOCX paragraph elements.

        Args:
            paragraph: DOCX paragraph object
            opts: DOCX partitioner options containing document info

        Yields:
            Image elements with metadata
        """
        if paragraph is None:
            return

        imgs = paragraph._element.xpath(".//pic:pic")
        if imgs:
            img_output_dir = opts.metadata_filename or "extracted_images"
            os.makedirs(img_output_dir, exist_ok=True)

            for img in imgs:
                try:
                    embed = img.xpath(".//a:blip/@r:embed")[0]
                    related_part = opts.document.part.related_parts[embed]
                    image_blob = related_part.blob
                    image = PILImage.open(io.BytesIO(image_blob))

                    # Generate unique image filename
                    image_filename = f"{embed}_{related_part.sha1}.png"
                    image_path = os.path.join(img_output_dir, image_filename)

                    # Save image
                    image.save(image_path)

                    # Create metadata
                    element_metadata = ElementMetadata(image_path=image_path)
                    yield Image(text="IMAGE", metadata=element_metadata)
                except Exception as e:
                    print(f"Warning: Failed to extract image from DOCX: {e}")
                    continue


def ensure_directory(path: str) -> str:
    """
    Ensure a directory exists, create if it doesn't.

    Args:
        path: Directory path

    Returns:
        Absolute path to the directory
    """
    abs_path = os.path.abspath(path)
    os.makedirs(abs_path, exist_ok=True)
    return abs_path


def get_file_extension(file_path: str) -> str:
    """
    Get the file extension in lowercase.

    Args:
        file_path: Path to file

    Returns:
        File extension (e.g., 'pdf', 'docx', 'txt')
    """
    return Path(file_path).suffix.lower().lstrip(".")


def is_supported_file(file_path: str) -> bool:
    """
    Check if file type is supported.

    Args:
        file_path: Path to file

    Returns:
        True if file type is supported
    """
    supported_extensions = {
        "txt",
        "pdf",
        "docx",
        "doc",
        "pptx",
        "ppt",
        "xlsx",
        "xls",
        "html",
        "htm",
        "xml",
        "md",
        "rst",
    }
    ext = get_file_extension(file_path)
    return ext in supported_extensions
