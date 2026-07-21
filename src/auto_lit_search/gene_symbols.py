"""Pure helpers for UniProt/MyGene gene symbols and alias lists."""

from __future__ import annotations

import re
from typing import Any, Iterable, List, Optional


def normalize_alias_list(aliases: Optional[Iterable[Any]]) -> List[str]:
    """Dedupe alias tokens case-insensitively; preserve first-seen casing."""
    out: List[str] = []
    seen: set[str] = set()
    if not aliases:
        return out
    for raw in aliases:
        if raw is None:
            continue
        s = str(raw).strip()
        if not s or s == "-" or s.lower() == "nan":
            continue
        lk = s.lower()
        if lk in seen:
            continue
        seen.add(lk)
        out.append(s)
    return out


def format_gene_aliases(aliases: Optional[Iterable[Any]]) -> Optional[str]:
    """Pipe-join sorted unique aliases for idmap CSV storage."""
    cleaned = normalize_alias_list(aliases)
    if not cleaned:
        return None
    return "|".join(sorted(cleaned, key=str.lower))


def parse_gene_aliases(value: Any) -> List[str]:
    """Parse idmap ``gene_aliases`` cell (pipe/semicolon separated)."""
    if value is None:
        return []
    try:
        import pandas as pd

        if isinstance(value, float) and pd.isna(value):
            return []
    except Exception:
        pass
    if isinstance(value, (list, tuple)):
        return normalize_alias_list(value)
    s = str(value).strip()
    if not s or s.lower() == "nan":
        return []
    parts = re.split(r"[|;]", s)
    return normalize_alias_list(parts)


def aliases_excluding_primary(aliases: Optional[Iterable[Any]], primary: Optional[str]) -> List[str]:
    prim = (primary or "").strip().lower() if primary else ""
    out: List[str] = []
    for a in normalize_alias_list(aliases):
        if prim and a.lower() == prim:
            continue
        out.append(a)
    return out


def prefer_gene_name_uniprot(uniprot_name: Optional[str], mygene_name: Optional[str]) -> Optional[str]:
    """Prefer UniProt-recommended gene symbol whenever present."""
    ex = (uniprot_name or "").strip() if uniprot_name else ""
    mg = (mygene_name or "").strip() if mygene_name else ""
    if ex:
        return ex
    return mg or None


def gene_aliases_from_uniprot_gene(g0: dict) -> List[str]:
    """UniProt gene synonyms + ORF names (symbol-like aliases only)."""
    aliases: List[str] = []
    for key in ("synonyms", "orfNames"):
        for item in g0.get(key) or []:
            if isinstance(item, dict):
                val = item.get("value")
            else:
                val = item
            if isinstance(val, str) and val.strip():
                aliases.append(val.strip())
    return normalize_alias_list(aliases)


_ROMAN_NUMERALS = frozenset(
    {"I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X"}
)

# Trailing subcellular-localization qualifier UniProt appends after a comma
# (e.g. "..., mitochondrial"). Not part of the gene symbol.
_LOCALIZATION_SUFFIX_RE = re.compile(r",\s*[a-z][a-z-]*\s*$")

# Description head-nouns that are never a gene symbol even as the final token.
_GENERIC_PROTEIN_NAME_ENDS = frozenset(
    {
        "protein", "kinase", "hydrolase", "transferase", "domain", "family",
        "system", "factor", "membrane", "synthase", "peptidase", "lipase",
        "ligase", "reductase", "dehydrogenase", "oxidase", "subunit",
        "component", "type", "chain",
        "atpase", "homodimer",
    }
)


def is_symbol_like_token(token: Optional[str]) -> bool:
    """
    True only for tokens that look like a gene symbol rather than an English or
    biochemistry description word.

    Accepts: tokens with a digit (CG3861, Mettl5, RFT1), tokens with an internal
    capital (IscU, ApepP, RomA), and short all-caps symbols (PCNA, ATIC). Rejects
    bare lowercase words (homolog, helicase) and Capitalized dictionary words
    (Methyltransferase, Oxidoreductase), which are the common failure mode.
    """
    s = (token or "").strip()
    if len(s) < 3 or len(s) > 24:
        return False
    if not re.match(r"^[A-Za-z][A-Za-z0-9\-]*$", s):
        return False
    if s.lower() in _GENERIC_PROTEIN_NAME_ENDS:
        return False
    if any(ch.isdigit() for ch in s):
        return True
    has_lower = any(ch.islower() for ch in s)
    if has_lower and any(ch.isupper() for ch in s[1:]):
        return True
    if s.isupper() and s not in _ROMAN_NUMERALS:
        return True
    return False


def first_uniprot_orf_name(g0: dict) -> Optional[str]:
    """First ORF name (e.g. Drosophila CG#####) — a curated, literature-usable symbol."""
    for item in g0.get("orfNames") or []:
        val = item.get("value") if isinstance(item, dict) else item
        if isinstance(val, str) and val.strip():
            return val.strip()
    return None


def infer_gene_name_from_protein_description(entry: dict) -> Optional[str]:
    """
    Last-resort gene symbol recovery for entries lacking genes[].geneName and ORF /
    ordered-locus names. Some effectors carry the symbol as the final token of the
    protein name (e.g. "...transferase Lgt1" -> Lgt1). Only accept that token when it
    is symbol-like; a bare description word (homolog, mitochondrial, Methyltransferase)
    is worse than falling back to the locus tag, so those are rejected here.
    """
    pd = entry.get("proteinDescription") or {}
    candidates: List[str] = []
    rec = pd.get("recommendedName") or {}
    if (rec.get("fullName") or {}).get("value"):
        candidates.append(rec["fullName"]["value"])
    for sn in pd.get("submissionNames") or []:
        val = (sn.get("fullName") or {}).get("value")
        if val:
            candidates.append(val)
    for alt in pd.get("alternativeNames") or []:
        val = (alt.get("fullName") or {}).get("value")
        if val:
            candidates.append(val)
    for text in candidates:
        cleaned = _LOCALIZATION_SUFFIX_RE.sub("", text.strip())
        parts = cleaned.split()
        if not parts:
            continue
        last = parts[-1].strip(".,;")
        if last.lower() in _GENERIC_PROTEIN_NAME_ENDS:
            continue
        if is_symbol_like_token(last):
            return last
    return None


def collect_mygene_aliases(result: dict) -> List[str]:
    """Symbol + alias tokens only (never MyGene ``name`` / full-name prose)."""
    aliases: List[str] = []
    sym = result.get("symbol")
    if isinstance(sym, str) and sym.strip():
        aliases.append(sym.strip())
    alias = result.get("alias")
    if isinstance(alias, list):
        for x in alias:
            if isinstance(x, str) and x.strip():
                aliases.append(x.strip())
    elif isinstance(alias, str) and alias.strip():
        for part in re.split(r"[,;|]", alias):
            if part.strip():
                aliases.append(part.strip())
    return normalize_alias_list(aliases)
