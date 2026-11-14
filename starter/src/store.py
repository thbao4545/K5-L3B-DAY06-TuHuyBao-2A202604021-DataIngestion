"""Store and search vectors using PostgreSQL + pgvector."""
import json
import re
from typing import Any

import psycopg

from .models import Chunk

_FILTER_KEY = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")


def _vec(v: list[float]) -> str:
    return "[" + ",".join(repr(float(x)) for x in v) + "]"


class PgVectorStore:
    def __init__(self, database_url: str, dimensions: int) -> None:
        if not database_url:
            raise ValueError("database_url is required")
        if not isinstance(dimensions, int) or not 0 < dimensions <= 2000:
            raise ValueError("dimensions must be an int in 1..2000")
        self.database_url = database_url
        self.dimensions = dimensions

    def _connect(self):
        return psycopg.connect(self.database_url)

    def create_schema(self) -> None:
        with self._connect() as conn:
            conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            conn.execute(f"""
                CREATE TABLE IF NOT EXISTS chunks (
                    strategy    text NOT NULL,
                    chunk_id    text NOT NULL,
                    document_id text NOT NULL,
                    source      text NOT NULL,
                    page        integer NOT NULL,
                    text        text NOT NULL,
                    metadata    jsonb NOT NULL DEFAULT '{{}}',
                    embedding   vector({self.dimensions}) NOT NULL,
                    PRIMARY KEY (strategy, chunk_id)
                )""")
            conn.execute("CREATE INDEX IF NOT EXISTS chunks_meta_idx ON chunks USING gin (metadata)")
            conn.execute("CREATE INDEX IF NOT EXISTS chunks_doc_idx ON chunks (strategy, document_id)")

    def upsert(self, strategy: str, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        """Idempotent write; removes stale chunks of the re-ingested documents."""
        if len(chunks) != len(vectors):
            raise ValueError(f"{len(chunks)} chunks but {len(vectors)} vectors")
        for v in vectors:
            if len(v) != self.dimensions:
                raise ValueError(f"vector has {len(v)} dims, expected {self.dimensions}")
        if not chunks:
            return
        doc_ids = sorted({c.document_id for c in chunks})
        keep = [c.chunk_id for c in chunks]
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute("DELETE FROM chunks WHERE strategy = %s AND document_id = ANY(%s) "
                        "AND chunk_id <> ALL(%s)", (strategy, doc_ids, keep))
            cur.executemany(
                """INSERT INTO chunks (strategy, chunk_id, document_id, source, page, text, metadata, embedding)
                   VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb, %s::vector)
                   ON CONFLICT (strategy, chunk_id) DO UPDATE SET
                     document_id = EXCLUDED.document_id, source = EXCLUDED.source,
                     page = EXCLUDED.page, text = EXCLUDED.text,
                     metadata = EXCLUDED.metadata, embedding = EXCLUDED.embedding""",
                [(strategy, c.chunk_id, c.document_id, c.source, c.page, c.text,
                  json.dumps(c.metadata), _vec(v)) for c, v in zip(chunks, vectors)])

    def search(self, strategy: str, query_vector: list[float], top_k: int = 3,
               filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Cosine-distance top-k with SQL-side metadata filtering (values parameterized)."""
        if len(query_vector) != self.dimensions:
            raise ValueError(f"query has {len(query_vector)} dims, expected {self.dimensions}")
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        vec = _vec(query_vector)
        where, params = ["strategy = %s"], [vec, strategy]
        for key, value in (filters or {}).items():
            if not _FILTER_KEY.match(key):
                raise ValueError(f"invalid filter key: {key!r}")
            where.append("metadata @> %s::jsonb")
            params.append(json.dumps({key: value}))
        params.extend([vec, top_k])
        sql = f"""SELECT chunk_id, document_id, source, page, text, metadata,
                         1 - (embedding <=> %s::vector) AS score
                  FROM chunks WHERE {' AND '.join(where)}
                  ORDER BY embedding <=> %s::vector LIMIT %s"""
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(sql, params)
            return [{"chunk_id": r[0], "document_id": r[1], "source": r[2], "page": r[3],
                     "text": r[4], "metadata": r[5], "score": float(r[6])}
                    for r in cur.fetchall()]
