"""
Shared filters for literature search terms (mapping + Europe PMC pass2).

Ambiguous NCBI descriptions (e.g. FlyBase gene kdn → "knockdown") and short
housekeeping symbols (e.g. gap, CS) otherwise flood pass2 with off-target papers.
"""

from __future__ import annotations

import re
from typing import Optional

# Technique / method words that must never be used as gene search terms.
_METHOD_BLOCKLIST = frozenset(
    {
        "knockdown",
        "knock-down",
        "knock down",
        "knockout",
        "knock-out",
        "knock out",
        "overexpression",
        "over-expression",
        "rnai",
        "sirna",
        "shrna",
        "crispr",
        "mutant",
        "mutation",
        "transgenic",
        "rescue",
        "silencing",
        "depletion",
        "ablation",
    }
)

# Short all-letter symbols that match huge unrelated literature when searched alone.
_SHORT_AMBIGUOUS_SYMBOLS = frozenset(
    {
        "cs",
        "gap",
        "tpm",
        "eno",
        "ack",
        "act",
        "map",
        "cap",
        "cat",
        "cad",
        "can",
        "age",
        "set",
        "men",
        "max",
        "min",
        "end",
        "run",
        "out",
        "all",
        "and",
        "the",
        "for",
        "not",
        "dna",
        "rna",
        "atp",
        "gtp",
        "nad",
        "coa",
    }
)

_GENERIC_PHRASE_RE = re.compile(
    r"(?i)\b("
    r"family\s+protein|uncharacterized|hypothetical|"
    r"domain[- ]containing|conserved\s+protein|"
    r"putative\b.+\bprotein|\bprotein\b.+\bprotein"
    r")\b"
)

_ENDS_PROTEIN_RE = re.compile(r"(?i)\b(protein|enzyme|subunit|homolog|homologue)\s*$")

_TECHNICAL_ALIAS_RE = re.compile(
    r"(?i)(?:"
    r"^\d+$|"            # bare numeric database alias
    r".+_at$|"           # expression-array probe ID
    r"^anon[-_:]|"       # anonymous EST/clone designation
    r"^clone(?:\s|$)|"
    r"^orf\d+$"
    r")"
)

# Gene-like: mixed alphanumeric, or mixed case token (vipD, legK2, CG3861).
_GENE_LIKE_RE = re.compile(
    r"^(?=.*\d)[A-Za-z][A-Za-z0-9_.\-]{1,119}$|"
    r"^(?=.*[a-z])(?=.*[A-Z])[A-Za-z][A-Za-z0-9_.\-]{1,119}$|"
    r"^CG\d+$|"
    r"^Dmel[_\\]?CG\d+$|"
    r"^WD_?\d+$|"
    r"^[A-Z]{2,5}\d{2,}(_RS\d+)?$"  # locus-ish AVR58_RS15295 already handled elsewhere
)


def normalize_search_term(term: Optional[str]) -> Optional[str]:
    if term is None:
        return None
    s = str(term).strip()
    if not s or s.lower() == "nan" or s == "-":
        return None
    return s


def is_method_word(term: str) -> bool:
    return term.strip().lower() in _METHOD_BLOCKLIST


def is_generic_protein_phrase(term: str) -> bool:
    s = term.strip()
    if not s:
        return False
    if _GENERIC_PHRASE_RE.search(s):
        return True
    # Multi-word descriptions that end with protein/enzyme/etc.
    if " " in s and _ENDS_PROTEIN_RE.search(s):
        return True
    # Long free-text descriptions (NCBI gene description style).
    words = s.split()
    if len(words) >= 3 and not _GENE_LIKE_RE.match(s.replace(" ", "")):
        return True
    return False


def is_gene_like_symbol(term: str) -> bool:
    s = term.strip()
    if not s or " " in s:
        return False
    return bool(_GENE_LIKE_RE.match(s))


def is_short_ambiguous_symbol(term: str) -> bool:
    s = term.strip()
    if not s or " " in s:
        return False
    if is_gene_like_symbol(s):
        return False
    low = s.lower()
    if low in _SHORT_AMBIGUOUS_SYMBOLS:
        return True
    # Bare 1–2 letter tokens (CS, etc.)
    if len(s) < 3 and s.isalpha():
        return True
    return False


def is_technical_alias(term: str) -> bool:
    """True for probe, clone, EST, and other technical identifiers."""
    return bool(_TECHNICAL_ALIAS_RE.search(term.strip()))


def is_usable_search_term(
    term: Optional[str],
    *,
    kind: str = "synonym",
) -> bool:
    """
    Return True if ``term`` is safe to include in Europe PMC pass2 text search.

    ``kind`` is one of: gene_name, common_name, synonym, locus_tag, genbank_acc,
    or other (treated like synonym).
    """
    s = normalize_search_term(term)
    if not s:
        return False
    if len(s) < 2 or len(s) > 120:
        return False

    kind_l = (kind or "synonym").strip().lower()

    # Accessions / locus tags: only basic shape checks (caller already sanitizes junk).
    if kind_l in {"locus_tag", "genbank_acc", "genbank_acc_stem"}:
        return not s.isdigit()

    if is_method_word(s):
        return False

    if kind_l in {"alias", "synonym"} and is_technical_alias(s):
        return False

    if kind_l == "common_name":
        if is_generic_protein_phrase(s):
            return False
        if " " in s:
            # Prefer gene_name / curated short aliases over free-text descriptions.
            return False
        if is_short_ambiguous_symbol(s):
            return False
        return True

    # gene_name / synonym / default
    if is_generic_protein_phrase(s):
        return False
    if is_short_ambiguous_symbol(s):
        return False
    # Keep real short gene symbols (kdn, mip, vip) that are not in the ambiguous set.
    if len(s) < 3 and not is_gene_like_symbol(s):
        return False
    return True


def sanitize_common_name_for_idmap(common_name: Optional[str]) -> Optional[str]:
    """
    Keep display-oriented common names only when usable as search aliases.

    Method words and description phrases are dropped so idmap does not advertise
    them as searchable terms.
    """
    s = normalize_search_term(common_name)
    if not s:
        return None
    if is_usable_search_term(s, kind="common_name"):
        return s
    return None


def filter_terms(
    terms: list[str],
    *,
    kind: str = "synonym",
) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for t in terms:
        s = normalize_search_term(t)
        if not s:
            continue
        if not is_usable_search_term(s, kind=kind):
            continue
        lk = s.lower()
        if lk in seen:
            continue
        seen.add(lk)
        out.append(s)
    return out


def term_reject_reasons(term: Optional[str], *, kind: str = "synonym") -> list[str]:
    """Diagnostic flags for audit scripts."""
    s = normalize_search_term(term)
    if not s:
        return ["empty"]
    reasons: list[str] = []
    if len(s) < 2:
        reasons.append("too_short")
    if len(s) > 120:
        reasons.append("too_long")
    if is_method_word(s):
        reasons.append("method_word")
    if is_generic_protein_phrase(s):
        reasons.append("generic_protein_phrase")
    if is_short_ambiguous_symbol(s):
        reasons.append("short_ambiguous_symbol")
    kind_l = (kind or "synonym").strip().lower()
    if kind_l in {"alias", "synonym"} and is_technical_alias(s):
        reasons.append("technical_alias")
    if kind_l == "common_name" and " " in s:
        reasons.append("multiword_common_name")
    if len(s) < 3 and kind_l not in {"locus_tag", "genbank_acc", "genbank_acc_stem"}:
        if not is_gene_like_symbol(s) and "short_ambiguous_symbol" not in reasons:
            reasons.append("len_lt_3")
    if not reasons and not is_usable_search_term(s, kind=kind):
        reasons.append("rejected")
    return reasons
