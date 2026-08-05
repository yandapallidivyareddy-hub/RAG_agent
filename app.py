import os
import uvicorn

from fastapi import FastAPI
from langserve import add_routes

from pydantic import BaseModel, Field

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS


# =====================================================
# 1. Gemini API
# =====================================================

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0,
)

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    google_api_key=GOOGLE_API_KEY,
)

# =====================================================
# 2. Knowledge Base
# =====================================================

big_paragraph = """
The Internet is a global system of interconnected computer networks
that communicate using the TCP/IP protocol suite.

The origins of the Internet trace back to ARPANET, a research network
developed in the late 1960s by the U.S. Department of Defense.

Today the Internet supports communication, education,
business, entertainment, cloud computing, and artificial intelligence.
"""

documents = [Document(page_content=big_paragraph)]

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
)

docs = splitter.split_documents(documents)

vectorstore = FAISS.from_documents(
    docs,
    embeddings,
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# =====================================================
# 3. Prompt
# =====================================================

prompt = ChatPromptTemplate.from_template(
    """
Answer ONLY using the context below.

Context:
{context}

Question:
{question}
"""
)

rag_chain = (
    {
        "context": retriever,
        "question": RunnablePassthrough(),
    }
    | prompt
    | llm
    | StrOutputParser()
)

# =====================================================
# 4. Input Schema
# =====================================================

from langchain_core.runnables import RunnableLambda

class RAGInput(BaseModel):
    input: str = Field(description="User Question")


runnable = (
    RunnableLambda(lambda x: x["input"])
    | rag_chain
).with_types(
    input_type=RAGInput,
    output_type=str,
)

# =====================================================
# 5. FastAPI
# =====================================================

app = FastAPI(
    title="RAG API",
    version="1.0",
)

add_routes(
    app,
    runnable,
    path="/rag",
    playground_type="default",
)

# =====================================================
# 6. Run
# =====================================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
