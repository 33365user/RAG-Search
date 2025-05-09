This RAG (Retrieval-Augmented Generation) app is a simple web interface that lets you ask questions about files on your computer, such as PDFs, Word documents, or text files.

It reads and processes these files, breaking them into chunks and converting them into vector representations stored in a vector database (vectorstore), which enables fast similarity-based searching.

When you ask a question, the app retrieves the most relevant chunks from the vectorstore and then uses Mistral, a local AI model running via Ollama, to generate a helpful, grounded answer based on that information.

This project requires the following 3rd party downloads:

- Ollama (https://ollama.com/)
- Mistral (https://ollama.com/library/mistral)
- Python (https://www.python.org/downloads/)
