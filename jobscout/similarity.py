from __future__ import annotations

import math
import re
from collections import Counter
from typing import Iterable, Mapping

_TOKEN_RE = re.compile(r"[a-z][a-z0-9+#.\-]{1,}")
_STOPWORDS = frozenset(
    """a about above after all also an and any are as at be been being but by can could do does for
    from had has have he her his how i if in into is it its more most my no not of on or our out over
    she so some such than that the their them then there these they this to up us was we were what when
    where which while who will with within without would you your experience years work working team
    teams role roles company strong ability including etc using use used new""".split()
)


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens (keeps c++, c#, node.js-style terms), minus stopwords."""
    return [
        t.strip(".-")
        for t in _TOKEN_RE.findall((text or "").lower())
        if t.strip(".-") not in _STOPWORDS and len(t.strip(".-")) > 1
    ]


def _tfidf(counts: Counter[str], idf: Mapping[str, float]) -> dict[str, float]:
    vec = {t: (1.0 + math.log(c)) * idf[t] for t, c in counts.items() if t in idf}
    norm = math.sqrt(sum(v * v for v in vec.values()))
    return {t: v / norm for t, v in vec.items()} if norm else {}


def resume_alignment(docs: Mapping[str, str], resumes: Iterable[str]) -> dict[str, float]:
    """Score how well each posting's text aligns with the best-matching resume, 0..1.

    TF-IDF cosine similarity, with IDF taken from the postings themselves so terms common to every
    job description (and so uninformative) count for little. Each posting gets the best score across
    the supplied resumes, converted to its percentile within `docs` so the result is comparable
    across profiles and corpus sizes. Pure Python, no model or network.
    """
    resume_tokens = [Counter(tokenize(r)) for r in resumes if (r or "").strip()]
    if not docs or not resume_tokens:
        return {k: 0.0 for k in docs}

    doc_counts = {k: Counter(tokenize(v)) for k, v in docs.items()}
    df: Counter[str] = Counter()
    for c in doc_counts.values():
        df.update(c.keys())
    n = len(doc_counts)
    idf = {t: math.log((1 + n) / (1 + d)) + 1.0 for t, d in df.items()}

    resume_vecs = [_tfidf(c, idf) for c in resume_tokens]
    raw: dict[str, float] = {}
    for key, counts in doc_counts.items():
        dvec = _tfidf(counts, idf)
        raw[key] = max(
            (sum(w * rv.get(t, 0.0) for t, w in dvec.items()) for rv in resume_vecs), default=0.0
        )

    ordered = sorted(raw.values())
    if len(ordered) == 1:
        return {k: 1.0 if v > 0 else 0.0 for k, v in raw.items()}
    rank = {}
    for i, v in enumerate(ordered):
        rank.setdefault(v, i)  # ties share the lowest rank
    return {k: (rank[v] / (len(ordered) - 1) if v > 0 else 0.0) for k, v in raw.items()}
