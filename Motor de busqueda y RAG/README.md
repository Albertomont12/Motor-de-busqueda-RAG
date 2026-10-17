# Motor de Búsqueda y RAG

Aplicación local para subir documentos PDF y TXT, extraer y dividir su contenido en fragmentos, indexarlos como vectores y consultar los fragmentos relevantes con una pregunta en español.

## Requisitos

- Python 3.10 o posterior.
- Para generar respuestas con un LLM, configura OpenAI o instala Ollama. La aplicación también funciona sin ninguno de los dos: utiliza embeddings locales basados en hashing y devuelve los fragmentos recuperados, sin generar una respuesta conversacional.

## Ejecutar

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app:app --reload
```

Abre <http://127.0.0.1:8000>. La API interactiva está disponible en <http://127.0.0.1:8000/docs>.

## Proveedores de IA (opcionales)

**OpenAI:** establece `OPENAI_API_KEY` antes de iniciar el servidor. Por defecto usa `text-embedding-3-small` para embeddings y `gpt-4o-mini` para generar respuestas. Puedes cambiar esos valores con `OPENAI_EMBEDDING_MODEL` y `OPENAI_CHAT_MODEL`.

```powershell
$env:OPENAI_API_KEY = "tu-clave"
uvicorn app:app --reload
```

**Ollama:** inicia Ollama y define `OLLAMA_BASE_URL` (por ejemplo, `http://localhost:11434`). Por defecto utiliza `nomic-embed-text` para embeddings y `llama3.2` para respuestas; ambos modelos deben estar descargados. Puedes configurar `OLLAMA_EMBEDDING_MODEL` y `OLLAMA_CHAT_MODEL`. Si se configura Ollama, tiene prioridad sobre OpenAI.

Los documentos y sus vectores se guardan en `data/rag.sqlite3`. Para cambiar la ubicación, define `RAG_DATA_DIR`. Cada archivo puede tener hasta 20 MB; se admiten `.pdf` y `.txt` en UTF-8.

Los documentos deben indexarse con el mismo proveedor y modelo de embeddings que se usa al consultarlos. Al cambiar de proveedor/modelo, vuelve a subir los documentos para regenerar sus vectores.
