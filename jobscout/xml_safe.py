from __future__ import annotations

import re
import xml.etree.ElementTree as ET
import xml.parsers.expat

_DTD_OR_ENTITY_RE = re.compile(rb"<!(?:DOCTYPE|ENTITY)\b", re.I)


def _forbid_dtd(*_args: object, **_kwargs: object) -> None:
    raise ValueError("Refusing to parse XML containing DOCTYPE or ENTITY declarations")


def parse_xml_bytes(xml_bytes: bytes) -> ET.Element:
    """Parse untrusted XML bytes safely (uses defusedxml if installed, rejects DTD/ENTITY)."""
    try:
        import defusedxml.ElementTree as safe_et  # type: ignore

        return safe_et.fromstring(xml_bytes)
    except ImportError:
        pass

    if _DTD_OR_ENTITY_RE.search(xml_bytes):
        raise ValueError("Refusing to parse XML containing DOCTYPE or ENTITY declarations")

    # Enforce DTD/ENTITY rejection at the Expat level so UTF-16/UTF-32 and other encodings cannot bypass.
    expat = xml.parsers.expat.ParserCreate()
    expat.StartDoctypeDeclHandler = _forbid_dtd
    expat.EntityDeclHandler = _forbid_dtd
    expat.UnparsedEntityDeclHandler = _forbid_dtd
    expat.ExternalEntityRefHandler = _forbid_dtd
    try:
        expat.Parse(xml_bytes, True)
    except xml.parsers.expat.ExpatError as exc:
        raise ET.ParseError(str(exc)) from exc
    return ET.fromstring(xml_bytes)
