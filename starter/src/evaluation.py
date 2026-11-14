"""Evaluate retrieved evidence and preserve per-query traces."""
import re
from collections import defaultdict
from typing import Any


def _norm(s: str) -> str:
    return re.sub(r"\W+", " ", (s or "").lower()).strip()


def _golds(question: dict[str, Any]) -> list[dict[str, Any]]:
    """Primary gold plus documented alternatives in question['alt_gold']."""
    golds = [{"source": question["gold_source_id"], "page": question["gold_page"],
              "quote": question.get("evidence_quote", "")}]
    for alt in question.get("alt_gold", []):
        golds.append({"source": alt.get("gold_source_id", alt.get("source")),
                      "page": alt.get("gold_page", alt.get("page")),
                      "quote": alt.get("evidence_quote", alt.get("quote", ""))})
    return golds


def _evidence_covered(hit: dict[str, Any], gold: dict[str, Any], min_overlap: float = 0.6) -> bool:
    """Right source and page, and the chunk contains enough of the gold quote."""
    if hit.get("document_id") != gold["source"] and hit.get("source") != gold["source"]:
        return False
    if hit.get("page") != gold["page"]:
        return False
    quote = _norm(gold["quote"])
    if not quote:
        return True
    text = _norm(hit.get("text", ""))
    if quote in text:
        return True
    words, vocab = quote.split(), set(text.split())
    return sum(w in vocab for w in words) / len(words) >= min_overlap


def evaluate_query(question: dict[str, Any], hits: list[dict[str, Any]]) -> dict[str, Any]:
    """Hit@1 / Evidence Hit@3 trace. Coherence and grounding stay manual (None)."""
    golds = _golds(question)
    top = hits[:3]
    flags = [any(_evidence_covered(h, g) for g in golds) for h in top]
    flt = question.get("filter")
    return {
        "question_id": question["id"], "split": question.get("split"),
        "question": question["question"], "filter": flt, "gold": golds,
        "hit_at_1": bool(flags and flags[0]),
        "hit_at_3": any(flags),
        "filter_compliant": (all(h.get("metadata", {}).get(k) == v
                                 for h in top for k, v in flt.items()) if flt else None),
        "top3": [{"rank": i + 1, "chunk_id": h.get("chunk_id"), "source": h.get("source"),
                  "page": h.get("page"), "score": h.get("score"), "evidence_match": flags[i],
                  "metadata": h.get("metadata"), "text": h.get("text")}
                 for i, h in enumerate(top)],
        "manual": {"grounding": None, "coherence": None},
    }


def summarize(traces: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate Hit@1, Evidence Hit@3 and filter compliance by strategy and split."""
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for t in traces:
        strategy = t.get("strategy", "?")
        groups[(strategy, "all")].append(t)
        if t.get("split"):
            groups[(strategy, t["split"])].append(t)
    out: dict[str, Any] = {}
    for (strategy, split), items in sorted(groups.items()):
        n = len(items)
        h1, h3 = sum(t["hit_at_1"] for t in items), sum(t["hit_at_3"] for t in items)
        comp = [t["filter_compliant"] for t in items if t.get("filter_compliant") is not None]
        out.setdefault(strategy, {})[split] = {
            "n": n, "hit_at_1": h1, "hit_at_1_rate": h1 / n,
            "hit_at_3": h3, "hit_at_3_rate": h3 / n,
            "filter_compliance": f"{sum(comp)}/{len(comp)}" if comp else None,
        }
    return out
