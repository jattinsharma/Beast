"""Low-level SQLite storage for Beast's memory system.

Schema:
    memories(
        id INTEGER PRIMARY KEY,
        key TEXT NOT NULL,          -- canonical key (lowercased, trimmed)
        value TEXT NOT NULL,        -- the stored fact/value
        category TEXT DEFAULT 'personal',  -- personal / preference / reminder / general
        source TEXT,                -- how the fact was learned (user said, observed, etc.)
        confidence REAL DEFAULT 1.0,  -- confidence in accuracy (0.0 to 1.0)
        tags TEXT DEFAULT '',       -- comma-separated tags for filtering
        is_sensitive BOOLEAN DEFAULT FALSE,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )

Key design decisions:
- key is stored LOWERCASE for case-insensitive lookups
- upsert on duplicate key (UPDATE value + updated_at)
- full-text search via LIKE on key + value + tags
- separate from the agent's session state; survives restarts
"""

import logging
import os
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

logger = logging.getLogger("beast.memory")

# Default database location: beast/data/beast_memories.db
_DEFAULT_DB_DIR = Path(__file__).parent.parent / "data"
_DEFAULT_DB_PATH = _DEFAULT_DB_DIR / "beast_memories.db"

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS memories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    key TEXT NOT NULL,
    value TEXT NOT NULL,
    category TEXT NOT NULL DEFAULT 'personal',
    source TEXT,
    confidence REAL NOT NULL DEFAULT 1.0,
    tags TEXT NOT NULL DEFAULT '',
    is_sensitive INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_memories_key ON memories(key);
CREATE INDEX IF NOT EXISTS idx_memories_category ON memories(category);
"""


class MemoryStore:
    """Thread-safe SQLite-backed memory store."""

    def __init__(self, db_path: Optional[str] = None):
        """
        Args:
            db_path: Path to the SQLite database file. Defaults to
                     beast/data/beast_memories.db.
        """
        if db_path is None:
            db_path = str(_DEFAULT_DB_PATH)

        self._db_path = db_path
        self._local = threading.local()  # one connection per thread

        # Ensure the data directory exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)

        # Initialize schema on the main thread
        with self._connect() as conn:
            conn.executescript(SCHEMA_SQL)

        logger.info("[MEMORY] Store initialized at %s", db_path)

    @contextmanager
    def _connect(self):
        """Context manager yielding a thread-local connection."""
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self._db_path, timeout=10)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=5000")
            self._local.conn = conn
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def remember(self, key: str, value: str, category: str = "personal",
                 source: str = None, confidence: float = 1.0,
                 tags: str = "", is_sensitive: bool = False) -> int:
        """Store or update a memory. Returns the row id.

        If a memory with the same key already exists, its value, category,
        source, confidence, tags, is_sensitive, and updated_at are overwritten.
        """
        key_norm = key.lower().strip()
        with self._connect() as conn:
            # Check for existing
            row = conn.execute(
                "SELECT id FROM memories WHERE key = ?", (key_norm,)
            ).fetchone()
            if row:
                conn.execute(
                    "UPDATE memories SET value = ?, category = ?, source = ?, confidence = ?, "
                    "tags = ?, is_sensitive = ?, updated_at = CURRENT_TIMESTAMP "
                    "WHERE id = ?",
                    (value, category, source, confidence, tags, int(is_sensitive), row["id"]),
                )
                logger.info("[MEMORY] Updated: key=%r (id=%d)", key_norm, row["id"])
                return row["id"]
            else:
                cur = conn.execute(
                    "INSERT INTO memories (key, value, category, source, confidence, tags, is_sensitive) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (key_norm, value, category, source, confidence, tags, int(is_sensitive)),
                )
                logger.info("[MEMORY] Stored: key=%r (id=%d)", key_norm, cur.lastrowid)
                return cur.lastrowid

    def recall(self, key: str) -> Optional[dict]:
        """Look up a memory by exact key (case-insensitive). Returns dict or None."""
        key_norm = key.lower().strip()
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM memories WHERE key = ?", (key_norm,)
            ).fetchone()
            if row:
                return dict(row)
            return None

    def search(self, query: str, category: Optional[str] = None,
               limit: int = 10) -> list[dict]:
        """Search memories by partial match on key, value, or tags.

        Args:
            query: Search term (case-insensitive LIKE)
            category: Optional category filter
            limit: Max results to return
        """
        q = f"%{query.lower().strip()}%"
        with self._connect() as conn:
            if category:
                rows = conn.execute(
                    "SELECT * FROM memories WHERE (key LIKE ? OR value LIKE ? OR tags LIKE ?) "
                    "AND category = ? ORDER BY updated_at DESC LIMIT ?",
                    (q, q, q, category, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM memories WHERE (key LIKE ? OR value LIKE ? OR tags LIKE ?) "
                    "ORDER BY updated_at DESC LIMIT ?",
                    (q, q, q, limit),
                ).fetchall()
            return [dict(r) for r in rows]

    def forget(self, key: str) -> bool:
        """Delete a memory by key. Returns True if something was deleted."""
        key_norm = key.lower().strip()
        with self._connect() as conn:
            cur = conn.execute("DELETE FROM memories WHERE key = ?", (key_norm,))
            deleted = cur.rowcount > 0
            if deleted:
                logger.info("[MEMORY] Forgotten: key=%r", key_norm)
            else:
                logger.info("[MEMORY] Nothing to forget for key=%r", key_norm)
            return deleted

    def forget_by_id(self, memory_id: int) -> bool:
        """Delete a memory by id. Returns True if something was deleted."""
        with self._connect() as conn:
            cur = conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
            return cur.rowcount > 0

    def list_all(self, category: Optional[str] = None,
                 limit: int = 50) -> list[dict]:
        """List all stored memories, newest first."""
        with self._connect() as conn:
            if category:
                rows = conn.execute(
                    "SELECT * FROM memories WHERE category = ? "
                    "ORDER BY updated_at DESC LIMIT ?",
                    (category, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM memories ORDER BY updated_at DESC LIMIT ?",
                    (limit,),
                ).fetchall()
            return [dict(r) for r in rows]

    def count(self, category: Optional[str] = None) -> int:
        """Count stored memories."""
        with self._connect() as conn:
            if category:
                row = conn.execute(
                    "SELECT COUNT(*) as cnt FROM memories WHERE category = ?",
                    (category,),
                ).fetchone()
            else:
                row = conn.execute("SELECT COUNT(*) as cnt FROM memories").fetchone()
            return row["cnt"]

    def clear(self, category: Optional[str] = None) -> int:
        """Delete all memories (optionally within a category). Returns count deleted."""
        with self._connect() as conn:
            if category:
                cur = conn.execute(
                    "DELETE FROM memories WHERE category = ?", (category,)
                )
            else:
                cur = conn.execute("DELETE FROM memories")
            count = cur.rowcount
            logger.info("[MEMORY] Cleared %d memories (category=%r)", count, category)
            return count