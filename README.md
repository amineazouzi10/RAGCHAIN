# RAGCHAIN

![RAGCHAIN Banner](https://via.placeholder.com/728x200.png?text=RAGCHAIN)

> **RAGCHAIN** is a FastAPI application leveraging [LangChain Community](https://github.com/langchain-ai/langchain) to perform question-answering on uploaded PDF or text documents, using OpenAI for embeddings and chat. This project demonstrates a Retrieval-Augmented Generation (RAG) workflow, where you:  
> 1. **Ingest** a document and create embeddings  
> 2. **Retrieve** the most relevant chunks given a user query  
> 3. **Generate** a final response using a Large Language Model (LLM)

## Table of Contents

1. [Features](#features)  
2. [Prerequisites](#prerequisites)  
3. [Installation](#installation)  
4. [Running the Application](#running-the-application)  
5. [Usage](#usage)  
   - [Querying Documents](#querying-documents)  
6. [Project Structure](#project-structure)  

---

## Features

- **FastAPI** for quick, efficient REST endpoints  
- **LangChain Community** modules for text loading, PDF parsing, embeddings, and vector stores  
- **OpenAIEmbeddings** for semantic search  
- **RAG-based** question-answering flow  
- **Simple** file upload endpoint to handle `.pdf` or `.txt` documents  

## Prerequisites

- **Python 3.9+** (recommended)  
- [pip](https://pip.pypa.io/en/stable/installation/) or a similar package manager  
- An [OpenAI API Key](https://platform.openai.com/account/api-keys)

## Installation

1. **Clone or Download** the repository:

   ```bash
   git clone https://github.com/yourusername/RAGCHAIN.git
   cd RAGCHAIN
2. **Install Dependencies**:

```bash
pip install -r requirements.txt
```
3. **Create and configure your .env file**:
```bash
echo "OPENAI_API_KEY=your-api-key-here" > .env
```
- Replace your-api-key-here with your actual OpenAI API key.

## Running the Application

1. **Start the FastAPI server with Uvicorn**:
```bash
uvicorn ragsystem:app --host 0.0.0.0 --port 8000
```
- `ragsystem.py` is the main file containing the FastAPI `app`.
- `app` is the FastAPI application instance.

2. **Access the API Docs**:
- Open http://localhost:8000/docs in your browser.
- You’ll see the interactive FastAPI documentation with available endpoints.

3. **Check Environment Variables**:
- Ensure your environment variable `OPENAI_API_KEY` is loaded properly (e.g., via `.env`).
- If the key isn’t found, the application may raise an error.

## USAGE

- ## Querying Documents
- Endpoint: `POST /query`
- Method: `POST`
- **Form Fields**:
  - **`file`**: A `.pdf` or `.txt` document you want to upload and query against.
  - **`question`**: A string that represents your query about the document’s content.

**Example Using `cURL`**

```bash
curl.exe -X POST `
    -F "file=@path/to/your/document.pdf" `
    -F "question=What is the main topic of this document?" `
    http://localhost:8000/query
```
- `file=@path/to/your/document.pdf`: Replace `path/to/your/document.pdf` with the actual path to your PDF or text file.
- `question=<your-question>`: You can provide any question relevant to the document.

**Response Example**
```json
{
  "response": "This document mainly discusses..."
}
```

**If an error occurs (like a missing file or a processing issue), you might receive:**
```json
{
  "error": "Error message"
}
```



Thought for a couple of seconds
markdown
Copy code
## Project Structure

```bash
RAGCHAIN/
├── ragsystem.py              # Main FastAPI application
├── requirements.txt          # Project dependencies
├── .env                      # Environment variables (ignored by Git)
├── .gitignore                # Ignores .env, __pycache__, etc.
└── README.md                 # Usage guide (this file)
```
**Explanation of Key Files**
- **ragsystem.py**
  - Contains all the logic for:
    - Loading documents (PDF or text)
    - Splitting text into chunks
    - Creating embeddings with OpenAI
    - Creating a vector store (Chroma) for similarity search
    - Generating responses via LangChain’s chain approach
    - Defining the `/query` endpoint with FastAPI
- **requirements.txt**
  - Lists the Python packages (and versions) your project depends on.
  Install them using:
```bash
pip install -r requirements.txt
```

- **.env**
  - Holds environment variables like `OPENAI_API_KEY`.


# RAGCHAIN




