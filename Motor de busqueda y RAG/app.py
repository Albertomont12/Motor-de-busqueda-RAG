import hashlib
import json
import math
import os
import re
import sqlite3
import struct
import urllib.error
import urllib.request
from contextlib import closing
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent
DATA_DIR = Path(os.getenv("RAG_DATA_DIR", ROOT / "data"))
DATABASE_PATH = DATA_DIR / "rag.sqlite3"
STATIC_DIR = ROOT / "static"
MAX_FILE_BYTES = 20 * 1024 * 1024
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 160
LOCAL_EMBEDDING_DIMENSIONS = 512
OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "").rstrip("/")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

app = FastAPI(title="Motor de Búsqueda y RAG", version="1.0.0")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class QuestionRequest(BaseModel):
    question: str = Field(min_length=2, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=10)


def current_embedding_model() -> str:
    if OLLAMA_URL:
        return f"ollama:{os.getenv('OLLAMA_EMBEDDING_MODEL', 'nomic-embed-text')}"
    if OPENAI_API_KEY:
        return f"openai:{os.getenv('OPENAI_EMBEDDING_MODEL', 'text-embedding-3-small')}"
    return "local:feature-hash-512"


def database() -> sqlite3.Connection:
    connection = sqlite3.connect(DATABASE_PATH, timeout=15)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def initialize_database() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with closing(database()) as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                chunk_index INTEGER NOT NULL,
                content TEXT NOT NULL,
                embedding TEXT NOT NULL,
                embedding_model TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id);
            """
        )
        connection.commit()


@app.on_event("startup")
def on_startup() -> None:
    initialize_database()


def api_request(url: str, payload: dict[str, Any], headers: dict[str, str] | None = None) -> dict[str, Any]:
    request_headers = {"Content-Type": "application/json"}
    request_headers.update(headers or {})
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=request_headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            return json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
        raise HTTPException(
            status_code=502,
            detail=f"No se pudo conectar con el proveedor de IA ({type(error).__name__}).",
        ) from error


def local_embedding(text: str) -> list[float]:
    vector = [0.0] * LOCAL_EMBEDDING_DIMENSIONS
    tokens = re.findall(r"\w+", text.lower(), flags=re.UNICODE)
    features = tokens + [f"{left}_{right}" for left, right in zip(tokens, tokens[1:])]
    for feature in features:
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest[:4], "little")
        index = value % LOCAL_EMBEDDING_DIMENSIONS
        vector[index] += 1.0 if digest[4] & 1 else -1.0
    magnitude = math.sqrt(sum(component * component for component in vector))
    return [component / magnitude for component in vector] if magnitude else vector


def embed_texts(texts: list[str]) -> list[list[float]]:
    if OLLAMA_URL:
        model = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
        result = api_request(
            f"{OLLAMA_URL}/api/embed",
            {"model": model, "input": texts},
        )
        embeddings = result.get("embeddings")
        if not isinstance(embeddings, list) or len(embeddings) != len(texts):
            raise HTTPException(status_code=502, detail="Ollama devolvió embeddings con un formato inesperado.")
        return embeddings
    if OPENAI_API_KEY:
        model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        result = api_request(
            "https://api.openai.com/v1/embeddings",
            {"model": model, "input": texts},
            {"Authorization": f"Bearer {OPENAI_API_KEY}"},
        )
        data = result.get("data")
        if not isinstance(data, list) or len(data) != len(texts):
            raise HTTPException(status_code=502, detail="OpenAI devolvió embeddings con un formato inesperado.")
        return [item["embedding"] for item in data]
    return [local_embedding(text) for text in texts]


def split_into_chunks(text: str) -> list[str]:
    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return []
    chunks = []
    start = 0
    while start < len(normalized):
        end = min(start + CHUNK_SIZE, len(normalized))
        if end < len(normalized):
            boundary = normalized.rfind(" ", start + CHUNK_SIZE // 2, end)
            if boundary > start:
                end = boundary
        chunk = normalized[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(normalized):
            break
        start = max(end - CHUNK_OVERLAP, start + 1)
    return chunks


async def extract_text(upload: UploadFile) -> tuple[str, bytes]:
    filename = Path(upload.filename or "").name
    extension = Path(filename).suffix.lower()
    if extension not in {".pdf", ".txt"}:
        raise HTTPException(status_code=415, detail="Formato no compatible. Sube un archivo PDF o TXT.")
    content = await upload.read(MAX_FILE_BYTES + 1)
    if not content:
        raise HTTPException(status_code=400, detail="El archivo está vacío.")
    if len(content) > MAX_FILE_BYTES:
        raise HTTPException(status_code=413, detail="El archivo supera el límite de 20 MB.")
    if extension == ".txt":
        try:
            text = content.decode("utf-8-sig")
        except UnicodeDecodeError as error:
            raise HTTPException(status_code=400, detail="El archivo TXT debe estar codificado en UTF-8.") from error
    else:
        try:
            from pypdf import PdfReader

            reader = PdfReader(content)
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
        except HTTPException:
            raise
        except Exception as error:
            raise HTTPException(status_code=400, detail="No se pudo leer el PDF. Comprueba que no esté dañado o protegido.") from error
    if not text.strip():
        raise HTTPException(status_code=400, detail="No se encontró texto extraíble en el documento.")
    return text, content


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        return 0.0
    return sum(a * b for a, b in zip(left, right))


def generate_answer(question: str, sources: list[dict[str, Any]]) -> str:
    context = "\n\n".join(
        f"[{index}] {source['name']}: {source['content']}"
        for index, source in enumerate(sources, start=1)
    )
    if OLLAMA_URL:
        result = api_request(
            f"{OLLAMA_URL}/api/chat",
            {
                "model": os.getenv("OLLAMA_CHAT_MODEL", "llama3.2"),
                "stream": False,
                "messages": [
                    {
                        "role": "system",
                        "content": "Responde en español usando solo el contexto. Si no contiene la respuesta, dilo claramente. Cita las fuentes con [n].",
                    },
                    {"role": "user", "content": f"Contexto:\n{context}\n\nPregunta: {question}"},
                ],
            },
        )
        return result.get("message", {}).get("content", "El modelo local no devolvió una respuesta.")
    if OPENAI_API_KEY:
        result = api_request(
            "https://api.openai.com/v1/chat/completions",
            {
                "model": os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
                "messages": [
                    {
                        "role": "system",
                        "content": "Responde en español usando solo el contexto proporcionado. Si no contiene la respuesta, dilo claramente. Cita las fuentes con [n].",
                    },
                    {"role": "user", "content": f"Contexto:\n{context}\n\nPregunta: {question}"},
                ],
                "temperature": 0.2,
            },
            {"Authorization": f"Bearer {OPENAI_API_KEY}"},
        )
        choices = result.get("choices")
        if not choices:
            raise HTTPException(status_code=502, detail="OpenAI devolvió una respuesta sin contenido.")
        return choices[0]["message"]["content"]
    excerpts = "\n\n".join(
        f"[{index}] {source['content']}"
        for index, source in enumerate(sources, start=1)
    )
    return (
        "No hay un LLM configurado; estos son los fragmentos más relacionados que encontré. "
        "Configura OpenAI u Ollama para generar una respuesta conversacional.\n\n"
        f"{excerpts}"
    )


@app.get("/")
def home() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health() -> dict[str, str]:
    provider = "Ollama" if OLLAMA_URL else "OpenAI" if OPENAI_API_KEY else "Local"
    return {"status": "ok", "embedding_provider": provider}


@app.get("/api/documents")
def list_documents() -> list[dict[str, Any]]:
    with closing(database()) as connection:
        rows = connection.execute(
            """
            SELECT d.id, d.name, d.created_at, COUNT(c.id) AS chunks
            FROM documents d LEFT JOIN chunks c ON c.document_id = d.id
            GROUP BY d.id ORDER BY d.created_at DESC, d.id DESC
            """
        ).fetchall()
    return [dict(row) for row in rows]


@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...)) -> dict[str, Any]:
    text, raw_content = await extract_text(file)
    chunks = split_into_chunks(text)
    if not chunks:
        raise HTTPException(status_code=400, detail="No se pudo dividir el documento en fragmentos.")
    vectors = embed_texts(chunks)
    if any(not isinstance(vector, list) or not vector for vector in vectors):
        raise HTTPException(status_code=502, detail="El proveedor devolvió un embedding vacío.")
    model = current_embedding_model()
    name = Path(file.filename or "documento").name
    digest = hashlib.sha256(raw_content).hexdigest()
    try:
        with closing(database()) as connection:
            cursor = connection.execute(
                "INSERT INTO documents(name, content_hash) VALUES (?, ?)",
                (name, digest),
            )
            document_id = cursor.lastrowid
            connection.executemany(
                """
                INSERT INTO chunks(document_id, chunk_index, content, embedding, embedding_model)
                VALUES (?, ?, ?, ?, ?)
                """,
                [
                    (document_id, index, chunk, json.dumps(vector), model)
                    for index, (chunk, vector) in enumerate(zip(chunks, vectors))
                ],
            )
            connection.commit()
    except sqlite3.Error as error:
        raise HTTPException(status_code=500, detail="No se pudo guardar el documento en la base vectorial.") from error
    return {"id": document_id, "name": name, "chunks": len(chunks)}


@app.post("/api/ask")
def ask_question(request: QuestionRequest) -> dict[str, Any]:
    query_vector = embed_texts([request.question])[0]
    model = current_embedding_model()
    with closing(database()) as connection:
        rows = connection.execute(
            """
            SELECT c.content, c.embedding, d.name, d.id AS document_id, c.chunk_index
            FROM chunks c JOIN documents d ON d.id = c.document_id
            WHERE c.embedding_model = ?
            """,
            (model,),
        ).fetchall()
    if not rows:
        raise HTTPException(
            status_code=404,
            detail="No hay documentos compatibles indexados. Sube un documento o vuelve a indexarlo con el proveedor actual.",
        )
    ranked = sorted(
        (
            {
                "content": row["content"],
                "name": row["name"],
                "document_id": row["document_id"],
                "chunk_index": row["chunk_index"],
                "score": cosine_similarity(query_vector, json.loads(row["embedding"])),
            }
            for row in rows
        ),
        key=lambda item: item["score"],
        reverse=True,
    )[: request.top_k]
    answer = generate_answer(request.question, ranked)
    sources = [
        {
            "name": source["name"],
            "document_id": source["document_id"],
            "chunk_index": source["chunk_index"],
            "score": round(source["score"], 4),
            "content": source["content"],
        }
        for source in ranked
    ]
    return {"answer": answer, "sources": sources}


@app.delete("/api/documents/{document_id}")
def delete_document(document_id: int) -> dict[str, bool]:
    with closing(database()) as connection:
        cursor = connection.execute("DELETE FROM documents WHERE id = ?", (document_id,))
        connection.commit()
    if cursor.rowcount == 0:
        raise HTTPException(status_code=404, detail="No se encontró el documento.")
    return {"deleted": True}
