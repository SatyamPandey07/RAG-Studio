"""
Document Parser Service
Extracts plain text from uploaded files: PDF, TXT, MD, HTML, CSV, JSON, DOCX
Falls back to raw byte decoding for unknown types.
"""
import io
import json
import csv
import re
from typing import Tuple


class DocumentParser:

    @staticmethod
    def parse(filename: str, file_bytes: bytes) -> Tuple[str, dict]:
        """
        Returns (extracted_text, metadata_dict).
        metadata: word_count, char_count, language (heuristic), page_count (PDF)
        """
        ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else "txt"

        if ext == "pdf":
            text, meta = DocumentParser._parse_pdf(file_bytes)
        elif ext in ("txt", "md", "markdown", "rst"):
            text, meta = DocumentParser._parse_text(file_bytes)
        elif ext in ("html", "htm"):
            text, meta = DocumentParser._parse_html(file_bytes)
        elif ext == "csv":
            text, meta = DocumentParser._parse_csv(file_bytes)
        elif ext == "json":
            text, meta = DocumentParser._parse_json(file_bytes)
        elif ext == "docx":
            text, meta = DocumentParser._parse_docx(file_bytes)
        else:
            text, meta = DocumentParser._parse_text(file_bytes)

        word_count = len(text.split())
        meta["word_count"] = word_count
        meta["char_count"] = len(text)
        meta["language"] = DocumentParser._detect_language(text)
        return text, meta

    @staticmethod
    def _parse_pdf(data: bytes) -> Tuple[str, dict]:
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(stream=data, filetype="pdf")
            pages = []
            for page in doc:
                pages.append(page.get_text())
            text = "\n\n".join(pages)
            return text, {"page_count": len(doc), "source_type": "pdf"}
        except ImportError:
            # Fallback: try to decode bytes as text
            text = data.decode("utf-8", errors="ignore")
            return text, {"page_count": 0, "source_type": "pdf_fallback"}
        except Exception as e:
            return f"[PDF parse error: {str(e)}]", {"page_count": 0, "source_type": "pdf_error"}

    @staticmethod
    def _parse_text(data: bytes) -> Tuple[str, dict]:
        # Try UTF-8 first, then latin-1
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            text = data.decode("latin-1", errors="ignore")
        return text, {"source_type": "text"}

    @staticmethod
    def _parse_html(data: bytes) -> Tuple[str, dict]:
        try:
            from html.parser import HTMLParser

            class TextExtractor(HTMLParser):
                def __init__(self):
                    super().__init__()
                    self.result = []
                    self._skip = False

                def handle_starttag(self, tag, attrs):
                    if tag in ("script", "style", "head"):
                        self._skip = True

                def handle_endtag(self, tag):
                    if tag in ("script", "style", "head"):
                        self._skip = False

                def handle_data(self, data):
                    if not self._skip:
                        self.result.append(data)

            extractor = TextExtractor()
            extractor.feed(data.decode("utf-8", errors="ignore"))
            text = " ".join(extractor.result)
            # Collapse whitespace
            text = re.sub(r"\s+", " ", text).strip()
            return text, {"source_type": "html"}
        except Exception as e:
            return data.decode("utf-8", errors="ignore"), {"source_type": "html_raw"}

    @staticmethod
    def _parse_csv(data: bytes) -> Tuple[str, dict]:
        text_io = io.StringIO(data.decode("utf-8", errors="ignore"))
        reader = csv.reader(text_io)
        rows = list(reader)
        lines = [", ".join(row) for row in rows]
        text = "\n".join(lines)
        return text, {"source_type": "csv", "row_count": len(rows)}

    @staticmethod
    def _parse_json(data: bytes) -> Tuple[str, dict]:
        try:
            obj = json.loads(data.decode("utf-8", errors="ignore"))
            text = json.dumps(obj, indent=2)
        except json.JSONDecodeError:
            text = data.decode("utf-8", errors="ignore")
        return text, {"source_type": "json"}

    @staticmethod
    def _parse_docx(data: bytes) -> Tuple[str, dict]:
        try:
            import docx
            doc = docx.Document(io.BytesIO(data))
            paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
            text = "\n\n".join(paragraphs)
            return text, {"source_type": "docx", "paragraph_count": len(paragraphs)}
        except ImportError:
            return data.decode("utf-8", errors="ignore"), {"source_type": "docx_fallback"}
        except Exception as e:
            return f"[DOCX parse error: {str(e)}]", {"source_type": "docx_error"}

    @staticmethod
    def _detect_language(text: str) -> str:
        """Simple heuristic: check for common non-English characters."""
        sample = text[:500]
        # Very basic heuristic
        if not sample:
            return "en"
        ascii_ratio = sum(1 for c in sample if ord(c) < 128) / len(sample)
        return "en" if ascii_ratio > 0.85 else "other"
