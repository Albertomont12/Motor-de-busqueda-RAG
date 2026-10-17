La aplicación funciona en dos etapas: indexa tus documentos y luego busca en ellos para responder preguntas.
1.
Subes un PDF o TXT desde la página web. El servidor extrae el texto y lo divide en fragmentos superpuestos para conservar el contexto entre uno y otro.
2.
Genera un vector por fragmento —una representación numérica del texto— y guarda el fragmento, su vector y el nombre del archivo en data/rag.sqlite3.
3.
Escribes una pregunta. El sistema genera un vector para ella, lo compara con los vectores guardados mediante similitud coseno y selecciona los fragmentos más cercanos.
4.
Genera la respuesta con esos fragmentos como contexto. Con OpenAI u Ollama configurado, el modelo redacta una respuesta en español y puede citar las fuentes. Sin configurar un proveedor, muestra los fragmentos relevantes, pero no genera una respuesta conversacional.
La página web se sirve en http://127.0.0.1:8000; también puedes explorar la API en http://127.0.0.1:8000/docs. El código principal está en app.py, la interfaz en static/index.html y las instrucciones para configurar los proveedores están en README.md. La base vectorial es SQLite, así que no requiere instalar Pinecone o Milvus.
