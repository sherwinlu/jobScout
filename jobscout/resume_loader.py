from __future__ import annotations

import re
import zipfile
from pathlib import Path

from jobscout.xml_safe import parse_xml_bytes

WORD_NS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
_MAX_DOCX_XML_BYTES = 10 * 1024 * 1024
_MAX_DOCX_TOTAL_BYTES = 25 * 1024 * 1024


def extract_docx_text(path: Path) -> str:
    """Extract readable text from a .docx file using Python stdlib only."""
    paragraphs: list[str] = []
    total_bytes = 0
    with zipfile.ZipFile(path, "r") as zf:
        xml_names = sorted(
            (n for n in zf.namelist() if re.match(r"^word/(document|header\d*|footer\d*)\.xml$", n)),
            key=lambda n: (0 if n == "word/document.xml" else 1, n),
        )
        for name in xml_names:
            info = zf.getinfo(name)
            if info.file_size > _MAX_DOCX_XML_BYTES:
                raise ValueError(f"DOCX entry '{name}' exceeds size limit")
            total_bytes += info.file_size
            if total_bytes > _MAX_DOCX_TOTAL_BYTES:
                raise ValueError("DOCX total XML content exceeds size limit")
            try:
                raw_entry = zf.open(info, "r").read(_MAX_DOCX_XML_BYTES + 1)
                if len(raw_entry) > _MAX_DOCX_XML_BYTES:
                    raise ValueError(f"DOCX entry '{name}' exceeds decompressed size limit")
                root = parse_xml_bytes(raw_entry)
            except Exception:
                continue
            for para in root.iter(f"{WORD_NS}p"):
                line = "".join(node.text for node in para.iter(f"{WORD_NS}t") if node.text).strip()
                if line:
                    paragraphs.append(line)
    return "\n".join(paragraphs).strip()


def extract_pdf_text(path: Path) -> str:
    """Extract text from a PDF file using pypdf (optional dependency)."""
    try:
        from pypdf import PdfReader  # type: ignore

        reader = PdfReader(str(path))
        pages = [(page.extract_text() or "").strip() for page in reader.pages]
        return "\n".join(p for p in pages if p).strip()
    except ImportError:
        return ""
    except Exception:
        return ""


def load_resume_text(path: Path | None) -> str:
    """Read text from .md, .txt, .docx, or .pdf resume files."""
    if not path or not path.exists():
        return ""
    suffix = path.suffix.lower()
    if suffix in {".md", ".txt", ".markdown"}:
        return path.read_text(encoding="utf-8", errors="ignore").strip()
    if suffix == ".docx":
        try:
            return extract_docx_text(path)
        except Exception:
            return ""
    if suffix == ".pdf":
        return extract_pdf_text(path)
    return ""
