"""Day 06 pipeline: ingest -> chunk (A/B/C) -> embed -> store -> evaluate."""
import json
import os
import random
import time
from pathlib import Path

from dotenv import load_dotenv

from .chunking import by_paragraph, fixed_size
from .embeddings import MyEmbedder
from .evaluation import evaluate_query, summarize
from .ingestion import load_pdf
from .store import PgVectorStore

ROOT = Path(__file__).resolve().parents[1]
DATA, RESULTS = ROOT / "data", ROOT / "results"
BENCHMARK = DATA / "benchmark.json"  # list of question dicts (see BENCHMARK.md)
MIN_CONTENT_PAGES = 30
MIN_PAGE_CHARS = 200
FIXED_SIZE, FIXED_OVERLAP, PARA_MAX = 800, 100, 1000
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def main() -> None:
    load_dotenv(ROOT / ".env")
    db_url = os.getenv("DATABASE_URL", "")
    model = os.getenv("EMBEDDING_MODEL") or DEFAULT_MODEL

    # 1. corpus + benchmark
    pdfs = sorted(DATA.glob("*.pdf"))
    if not pdfs:
        raise SystemExit(f"No PDF in {DATA}")
    pages = []
    for pdf in pdfs:
        pages += load_pdf(pdf, document_id=pdf.stem, source=pdf.name)
    content = [p for p in pages if len(p.text) >= MIN_PAGE_CHARS]
    empty = [p.page for p in pages if p.metadata["empty"]]
    print(f"{len(pages)} pages, {len(content)} content pages, empty/needs OCR: {empty}")
    if len(content) < MIN_CONTENT_PAGES:
        raise SystemExit(f"Need >= {MIN_CONTENT_PAGES} content pages, found {len(content)}")
    if not BENCHMARK.exists():
        raise SystemExit(f"Write your questions to {BENCHMARK} first (id, split, question, "
                         "expected_answer, gold_source_id, gold_page, evidence_quote, filter?)")
    questions = json.loads(BENCHMARK.read_text(encoding="utf-8"))
    ids = [q["id"] for q in questions]
    if len(ids) != len(set(ids)):
        raise SystemExit("duplicate question ids")
    n_dev = sum(q["split"] == "dev" for q in questions)
    n_test = sum(q["split"] == "test" for q in questions)
    n_filter = sum(bool(q.get("filter")) for q in questions)
    if len(questions) < 20 or n_dev < 15 or n_test < 5 or n_filter < 3:
        raise SystemExit(f"Benchmark too small: {len(questions)} q, dev={n_dev}, "
                         f"test={n_test}, filter={n_filter}")

    # 2. strategies (C reuses B's chunks and differs by applying the metadata filter)
    chunk_sets = {
        "A_fixed": [c for p in content for c in fixed_size(p, FIXED_SIZE, FIXED_OVERLAP)],
        "B_paragraph": [c for p in content for c in by_paragraph(p, PARA_MAX)],
    }

    # 3. embed + store
    embedder = MyEmbedder(model)
    store = PgVectorStore(db_url, embedder.dimensions)
    store.create_schema()
    timings = {}
    for name, chunks in chunk_sets.items():
        t0 = time.perf_counter()
        store.upsert(name, chunks, embedder.embed_documents([c.text for c in chunks]))
        timings[name] = {"chunks": len(chunks), "seconds": round(time.perf_counter() - t0, 2)}
        print(name, timings[name])

    # 4. same questions on every strategy; full top-3 traces
    configs = [("A_fixed", "A_fixed", False), ("B_paragraph", "B_paragraph", False),
               ("C_paragraph_filter", "B_paragraph", True)]
    traces = []
    for q in questions:
        qv = embedder.embed_query(q["question"])
        for strategy, store_name, use_filter in configs:
            flt = q.get("filter") if use_filter else None
            t = evaluate_query(q, store.search(store_name, qv, top_k=3, filters=flt))
            t["strategy"], t["filter_applied"] = strategy, flt
            traces.append(t)

    # 5. report + fixed-seed samples for manual coherence/grounding scoring
    RESULTS.mkdir(exist_ok=True)
    summary = summarize(traces)
    rng = random.Random(6)
    test_ids = [q["id"] for q in questions if q["split"] == "test"]
    samples = {
        "grounding_questions": rng.sample(test_ids, min(5, len(test_ids))),
        "coherence_pages": sorted(rng.sample([p.page for p in content], min(10, len(content)))),
    }
    out = {"model": embedder.model_name, "dimensions": embedder.dimensions,
           "params": {"fixed": [FIXED_SIZE, FIXED_OVERLAP], "paragraph_max": PARA_MAX},
           "timings": timings, "summary": summary, "manual_samples": samples}
    (RESULTS / "summary.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    (RESULTS / "traces.json").write_text(json.dumps(traces, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Saved to {RESULTS}. Fill manual scores for: {samples}")


if __name__ == "__main__":
    main()
