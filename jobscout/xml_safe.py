from __future__ import annotations

import re
import xml.etree.ElementTree as ET

_DTD_OR_ENTITY_RE = re.compile(rb"<!(?:DOCTYPE|ENTITY)\b", re.I)


def parse_xml_bytes(xml_bytes: bytes) -> ET.Element:
    """Parse untrusted XML bytes safely (uses defusedxml if installed, rejects DTD/ENTITY)."""
    try:
        import defusedxml.ElementTree as safe_et  # type: ignore

        return safe_et.fromstring(xml_bytes)
    except ImportError:
        pass

    if _DTD_OR_ENTITY_RE.search(xml_bytes):
        raise ValueError("Refusing to parse XML containing DOCTYPE or ENTITY declarations")
    return ET.fromstring(xml_bytes)
