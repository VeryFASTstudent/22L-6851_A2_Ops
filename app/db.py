import sqlite3
from pathlib import Path


def _sqlite_path(url):
    if url.startswith("sqlite:///"):
        path = Path(url.removeprefix("sqlite:///"))
        path.parent.mkdir(parents=True, exist_ok=True)
        return str(path)
    return None


def connection(url):
    path = _sqlite_path(url)
    if path:
        conn = sqlite3.connect(path)
        conn.row_factory = sqlite3.Row
        return conn
    if url.startswith(("postgresql://", "postgres://")):
        import psycopg
        from psycopg.rows import dict_row
        return psycopg.connect(url, row_factory=dict_row)
    raise RuntimeError("DATABASE_URL must use sqlite:/// or postgresql://.")


def init_db(url):
    with connection(url) as conn:
        if url.startswith(("postgresql://", "postgres://")):
            conn.execute("""CREATE TABLE IF NOT EXISTS predictions (
                id BIGSERIAL PRIMARY KEY, filename TEXT NOT NULL, label TEXT NOT NULL,
                confidence DOUBLE PRECISION NOT NULL, model_name TEXT NOT NULL,
                model_version TEXT NOT NULL, created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
            )""")
        else:
            conn.execute("""CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT, filename TEXT NOT NULL,
                label TEXT NOT NULL, confidence REAL NOT NULL, model_name TEXT NOT NULL,
                model_version TEXT NOT NULL, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )""")
        conn.commit()


def add_prediction(url, filename, label, confidence, model_name, model_version):
    with connection(url) as conn:
        conn.execute(
            "INSERT INTO predictions(filename,label,confidence,model_name,model_version) VALUES(?,?,?,?,?)",
            (filename, label, confidence, model_name, model_version),
        )
        conn.commit()


def list_predictions(url, limit=100):
    with connection(url) as conn:
        return conn.execute(
            "SELECT * FROM predictions ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
