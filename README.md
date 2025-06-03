# AI Resume Search Engine

This project is an AI-powered resume/CV search system that helps you find relevant content snippets from a collection of CVs based on natural language queries. It leverages sentence embeddings and a vector database (Milvus) for highly efficient semantic search.

## Features

- **Smart search:** Enter a question or keywords and get the most relevant CV snippets instantly.
- **Flexible management:** Easily add new CVs in bulk; the system automatically processes, chunks, and stores data.
- **Result viewing & download:** Preview the most relevant snippets, see similarity scores, and download original CV files.
- **User-friendly interface:** Built on Streamlit for intuitive and fast search and management operations.

## Tech Stack

- **Python** (core logic)
- **Streamlit** (web interface)
- **sentence-transformers** (text embedding)
- **Milvus** (vector database for storing/searching embeddings)
- **LangChain** (text chunking)
- **pymilvus** (Milvus connector)

## How It Works

1. **Enter a query:** Users type a question or keyword (e.g., "python backend", "3 years project management experience").
2. **Query embedding:** The system generates an embedding for the input using a transformer model.
3. **Vector search:** The embedding is matched against all stored CV chunks in Milvus.
4. **Scoring:** Results are ranked by a weighted score combining similarity and chunk frequency per CV.
5. **Display:** Results show top CVs, most relevant snippets, and allow file download.

## Usage

### 1. Installation

```bash
git clone https://github.com/DucAnhishere/search_engine.git
cd search_engine
pip install -r requirements.txt
```

You need to install and start Milvus (see instructions: https://milvus.io/docs/install_standalone-docker.md).

### 2. Run the application

```bash
streamlit run src/app.py
```

### 3. Add new CVs

- Enter the folder path containing CVs in the sidebar and click "Process CVs".
- The system will automatically convert files, chunk and clean data, then store in the database.

### 4. Search for CVs

- Enter your query in the sidebar (e.g., "python backend", "project manager experience").
- View results, preview relevant snippets, and download CVs as needed.

## Who is this for?

- Recruiters and HR teams needing fast, semantic search across large CV collections.
- Any system managing and searching large document sets by content.

## Contributing

Pull requests and suggestions are welcome!

## License

Please add your license information here (if any).
