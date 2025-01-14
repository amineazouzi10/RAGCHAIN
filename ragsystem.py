import os
from dotenv import load_dotenv
from typing import List
from langchain.schema import Document
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_community.chat_models import ChatOpenAI  
from langchain_community.document_loaders import TextLoader  
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OpenAIEmbeddings  
from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts.chat import (
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain_community.vectorstores import Chroma  
from colorama import Fore
import uvicorn
from fastapi import File, UploadFile, Form


# Load environment variables from .env file
load_dotenv()

# Check for API key and provide a more helpful error message
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError(
        "OPENAI_API_KEY not found in environment variables. "
        "Please create a .env file in your project directory with your OpenAI API key:\n"
        "OPENAI_API_KEY=your-api-key-here"
    )

LANGUAGE_MODEL = "gpt-3.5-turbo-0125"

template = """
You are a customer support specialist. You assist users with general inquiries based on {context} and technical issues.
If you don't know the answer, you should invite the user to contact support by phone or email.
"""

system_message_prompt = SystemMessagePromptTemplate.from_template(template)
human_message_prompt = HumanMessagePromptTemplate.from_template("{question}")

chat_prompt_template = ChatPromptTemplate.from_messages(
    [system_message_prompt, human_message_prompt]
)

# Initialize model with the API key
model = ChatOpenAI(api_key=OPENAI_API_KEY, model_name=LANGUAGE_MODEL)



def load_documents(file_path: str, chunk_size: int = 1000, chunk_overlap: int = 100) -> List[Document]:
    """
    Load documents from either PDF or text files, split them into chunks, and prepare them for embedding.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    file_extension = os.path.splitext(file_path)[1].lower()
    
    if file_extension == '.pdf':
        loader = PyPDFLoader(file_path)
    elif file_extension == '.txt':
        loader = TextLoader(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_extension}. Only .pdf and .txt files are supported.")
    
    text_splitter = CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separator="\n"
    )
    
    try:

        documents = loader.load()
        chunks = text_splitter.split_documents(documents)      
        return chunks
    
    except Exception as e:
        raise Exception(f"Error processing document: {str(e)}")




def load_embeddings(documents: List[Document], user_query: str):
    """Create a vector store from a set of documents."""
    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
    db = Chroma.from_documents(documents, embeddings)
    docs = db.similarity_search(user_query)
    return db.as_retriever()



def generate_response(retriever, query: str) -> str:
    """Generate a response to a user query."""
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | chat_prompt_template
        | model
        | StrOutputParser()
    )

    return chain.invoke(query)



# Define the FastAPI app
app = FastAPI(
    title="LangChain API",
    version="1.0",
    description="An API server using LangChain's Runnable interfaces",
)

# Pydantic model for user queries
class QueryRequest(BaseModel):
    file_path: str
    question: str

if not os.path.exists("temp"):
    os.makedirs("temp")

@app.post("/query")
async def query_api(file: UploadFile = File(...), question: str = Form(...)):
    """
    Handle file uploads and process the user's question.
    """
    try:
        # Save the uploaded file to a temporary location
        file_path = f"temp/{file.filename}"
        with open(file_path, "wb") as f:
            f.write(await file.read())

        # Load and process the document
        documents = load_documents(file_path)
        retriever = load_embeddings(documents, question)
        response = generate_response(retriever, question)
        
        return {"response": response}

    except Exception as e:
        return {"error": str(e)}



if __name__ == "__main__":
    uvicorn.run("ragsystem:app", host="0.0.0.0", port=8000, log_level="info")