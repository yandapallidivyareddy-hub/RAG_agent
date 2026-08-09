
import os
import logging
import uvicorn

from fastapi import FastAPI
from langserve import add_routes
from pydantic import BaseModel, Field

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)

from langchain_text_splitters import RecursiveCharacterTextSplitter

# FAISS
from langchain_community.vectorstores import FAISS


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)


# ============================================================
# 1. GEMINI / GOOGLE GENAI CONFIGURATION
# ============================================================

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError(
        "GOOGLE_API_KEY environment variable is not set."
    )


# Use environment variables if available
MODEL_NAME = os.getenv(
    "MODEL_NAME",
    "gemini-3.5-flash"
)

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "gemini-embedding-001"
)


# ============================================================
# 2. INITIALIZE GEMINI LLM
# ============================================================

try:

    logger.info(
        "Initializing Gemini model=%s",
        MODEL_NAME
    )

    llm = ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        google_api_key=GOOGLE_API_KEY,
        temperature=0,
    )

except Exception as e:

    raise RuntimeError(
        f"Failed to initialize ChatGoogleGenerativeAI "
        f"with model={MODEL_NAME}: {e}"
    ) from e


# ============================================================
# 3. INITIALIZE GOOGLE EMBEDDINGS
# ============================================================

try:

    logger.info(
        "Initializing embedding model=%s",
        EMBEDDING_MODEL
    )

    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=GOOGLE_API_KEY,
    )

except Exception as e:

    raise RuntimeError(
        f"Failed to initialize GoogleGenerativeAIEmbeddings "
        f"with model={EMBEDDING_MODEL}: {e}"
    ) from e


# ============================================================
# 4. KNOWLEDGE BASE
# ============================================================

big_paragraph = """
Yandapalli Divya Reddy

24r21a05av@mlrit.ac.in | +91 8008801349 | Hyderabad

CAREER OBJECTIVE

Motivated Computer Science undergraduate skilled in Java, Python,
Web Development, and AI. Passionate about building innovative
web and AI-powered applications. Seeking software development
or AI/ML internships to apply my skills, solve real-world
problems, and grow as a developer.


WORK EXPERIENCE

AI-ML Intern
Global Next Consulting India Private Limited
Virtual
Mar 2026 - May 2026

1. Learned Machine Learning and Artificial Intelligence concepts.
2. Developed AI-based mini projects using Python.
3. Worked on real-world datasets for prediction problems.
4. Collaborated with mentors and team members.
5. Participated in project discussions and code reviews.
6. Improved problem-solving and debugging skills.
7. Gained exposure to Git, Python libraries, and AI workflows.


EDUCATION

B.Tech, Computer Science & Engineering
Marri Laxman Reddy Institute of Technology and Management
2024 - 2028

CGPA: 9.19/10

Practicing Data Structures and Algorithms.
Learning Java, Python, SQL, HTML, CSS, JavaScript.
Building academic and personal projects.
Participated in multiple hackathons.


Senior Secondary (XII)
Telangana State Board Of Intermediate Education
2024

Narayana Junior College

CGPA: 9.00/10


Secondary (X)
CBSE
2022

Hindu Public School

CGPA: 8.00/10


SKILLS

Java
Python
C Programming
SQL
MySQL
VS Code
Google Colab
Git
GitHub
Pandas
NumPy
Matplotlib
Seaborn
Streamlit
Data Analytics
Deep Learning
Data Visualization
Data Preprocessing
Problem Solving
Analytical Thinking
Presentation Skills


ADDITIONAL DETAILS

Secured 9.19 CGPA.

Participated in 3+ Hackathons.

Developed AI Job Readiness Prediction System.

Solved coding problems on HackerRank and CodeChef.

Delivered technical seminar presentations.

Worked on AI/ML internship projects.
"""


# ============================================================
# 5. CREATE DOCUMENT
# ============================================================

documents = [
    Document(page_content=big_paragraph)
]


# ============================================================
# 6. TEXT SPLITTING
# ============================================================

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
)

docs = splitter.split_documents(documents)

logger.info(
    "Created %d document chunks",
    len(docs)
)


# ============================================================
# 7. FAISS VECTOR STORE
# ============================================================

try:

    logger.info("Creating FAISS vector store...")

    vectorstore = FAISS.from_documents(
        docs,
        embeddings
    )

    logger.info(
        "FAISS vector store created successfully."
    )

except Exception as e:

    raise RuntimeError(
        f"Failed to initialize FAISS vector store: {e}"
    ) from e


# ============================================================
# 8. RETRIEVER
# ============================================================

retriever = vectorstore.as_retriever(
    search_kwargs={
        "k": 2
    }
)


# ============================================================
# 9. RAG PROMPT
# ============================================================

prompt = ChatPromptTemplate.from_template(
    """
You are a helpful assistant answering questions about
Yandapalli Divya Reddy.

Answer ONLY using the context provided below.

If the answer is not available in the context,
say:

"I don't have that information in my knowledge base."

Do not make up information.

Context:
{context}

Question:
{question}

Answer:
"""
)


# ============================================================
# 10. RAG CHAIN
# ============================================================

rag_chain = (
    {
        "context": retriever,
        "question": RunnablePassthrough(),
    }
    | prompt
    | llm
    | StrOutputParser()
)


# ============================================================
# 11. INPUT MODEL
# ============================================================

class RAGInput(BaseModel):

    input: str = Field(
        description="User Question"
    )


# ============================================================
# 12. INPUT EXTRACTOR
# ============================================================

def extract_input(x):

    logger.debug(
        "extract_input received type=%s value=%r",
        type(x),
        x
    )

    # Dictionary input
    if isinstance(x, dict):

        return (
            x.get("input")
            or x.get("question")
            or x.get("q")
            or x
        )

    # Pydantic model
    if hasattr(x, "input"):

        return x.input

    # String
    if isinstance(x, str):

        return x

    return x


# ============================================================
# 13. LANGSERVE CHAIN
# ============================================================

runnable = (
    RunnableLambda(extract_input)
    | rag_chain
).with_types(
    input_type=RAGInput,
    output_type=str,
)


# ============================================================
# 14. FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Divya Reddy RAG API",
    version="1.0.0",
)


# ============================================================
# 15. HEALTH CHECK
# ============================================================

@app.get("/")
def home():

    return {
        "message": "RAG API is running successfully!",
        "docs": "/docs",
        "playground": "/rag/playground",
        "invoke": "/rag/invoke",
    }


# ============================================================
# 16. LANGSERVE ROUTE
# ============================================================

add_routes(
    app,
    runnable,
    path="/rag",
    playground_type="default",
)


# ============================================================
# 17. RUN SERVER
# ============================================================

if __name__ == "__main__":

    port = int(
        os.getenv(
            "PORT",
            8000
        )
    )

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
    )

