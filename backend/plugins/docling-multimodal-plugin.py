from plugins.base import IngestPlugin, PluginRegistry
from typing import Dict, Any, List
from pathlib import Path
import os, tempfile, fitz
from docling.document_converter import DocumentConverter


@PluginRegistry.register
class DoclingMultimodalIngestPlugin(IngestPlugin):
    
    # Plugin metadata
    name = "docling_multimodal"
    kind = "file-ingest"
    description = "Ingests text, images and tables content using Docling and CLIP embeddings."
    supported_file_types = {"pdf"}

    def get_parameters(self) -> Dict[str, Dict[str, Any]]:
        return {
            "chunk_size": {
                "type": "integer",
                "default": 512,
                "description": "Mida de cada fragment de text si s'utilitza fragmentació per mida."
            },
            "chunk_strategy": {
                "type": "enum",
                "default": "semantic",
                "description": "Estrategia de fragmentació a utilitzar.",
                "options": ["semantic", "fixed", "layout", "image"]
            },
            "overlap": {
                "type": "integer",
                "default": 50,
                "description": "Nombre de tokens o unitats que es solapen entre fragments."
            },
            "ocr": {
                "type": "boolean",
                "default": True,
                "description": "Habilitar OCR per a imatges i PDFs que contenen text."
            },
            "include_images": {
                "type": "boolean",
                "default": True,
                "description": "Incloure imatges en els fragments generats."
            },
            "max_pages": {
                "type": "integer",
                "default": 100,
                "description": "Nombre màxim de pàgines a processar en un PDF."
            },
            "language":  {
                "type": "string",
                "default": "en",
                "description": "Idioma del contingut a processar. Utilitzat per a OCR i models de llenguatge."
            }
        }

    def ingest(self, file_path: str, **kwargs) -> List[Dict[str, Any]]:
        chunk_size = kwargs.get("chunk_size", 512)
        chunk_strategy = kwargs.get("chunk_strategy", "semantic")
        overlap = kwargs.get("overlap", 50)
        ocr = kwargs.get("ocr", True)
        include_images = kwargs.get("include_images", True)
        max_pages = kwargs.get("max_pages", 100)
        language = kwargs.get("language", "en")
        file_url = kwargs.get("file_url", "")

        chunks = []
        common_metadata = {
            "source": file_path,
            "modality": "multimodal",
            "embedding_hint": "clip",
            "converted_with": "Docling",
            "chunk_strategy": chunk_strategy,
            "chunk_size": chunk_size,
            "language": language,
            "ocr_enabled": ocr,
            "file_url": file_url,
            "original_filename": os.path.basename(file_path),
            "include_images": include_images,
            "overlap": overlap,
            "max_pages": max_pages
        }

        with open(file_path, "rb") as f:
            file_bytes = f.read()

        converter = DocumentConverter()

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
            temp_pdf.write(file_bytes)
            temp_pdf.flush()

            result = converter.convert(temp_pdf.name, ocr, chunk_size, overlap, chunk_strategy, max_pages, language)
            markdown = result.document.export_to_markdown()

        paragraphs = markdown.split("\n\n")

        for i in range(0, len(paragraphs), chunk_size):
            chunk_text = "\n\n".join(paragraphs[i:i + chunk_size])
            if not chunk_text.strip():
                continue

            chunk = {
                "text": chunk_text,
                "metadata": {
                    **common_metadata,
                    "type": "text",
                    "chunk_index": i // chunk_size
                }
            }
            chunks.append(chunk)
        
        if include_images:
            doc = fitz.open(file_path)
            
            for page_index in range(min(len(doc), max_pages)):
                page = doc[page_index]
                img_info = page.get_images(full=True)

                for img_index, img in enumerate(img_info):
                    xref = img[0]
                    base_img = doc.extract_image(xref)
                    img_bytes = base_img["image"]
                    ext = base_img["ext"]
                    filename = f"image_{page_index}_{img_index}.{ext}"
                    image_path = os.path.join("imatges", filename)

                    os.makedirs(os.path.dirname(image_path), exist_ok=True)
                    with open(image_path, "wb") as img:
                        img.write(img_bytes)

                    image_chunk = {
                        "text": image_path,
                        "metadata": {
                            **common_metadata,
                            "type": "imatge",
                            "image_path": image_path, 
                            "page_index": page_index,
                            "image_index": img_index
                        }
                    }
                    chunks.append(image_chunk)
        return chunks
