import os
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
from langchain_community.vectorstores import FAISS

# =====================================================
# 1. Gemini API
# =====================================================

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable is not set.")

llm = ChatGoogleGenerativeAI(
    model="models/gemini-2.5-flash",
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
Yandapalli Divya Reddy
24r21a05av@mlrit.ac.in | +91 8008801349 | Hyderabad
CAREER OBJECTIVE
Motivated Computer Science undergraduate skilled in Java, Python, Web Development, and AI. Passionate about building innovative
web and AI-powered applications. Seeking software development or AI/ML internships to apply my skills, solve real-world problems,
and grow as a developer.
WORK EXPERIENCE
AI-ML Intern • Internship Mar 2026 - May 2026
Global Next Consulting India Private Limited, Virtual
1. Learned Machine Learning and Artificial Intelligence concepts.
2. Developed AI-based mini projects using Python.
3. Worked on real-world datasets for prediction problems.
4. Collaborated with mentors and team members.
5. Participated in project discussions and code reviews.
6. Improved problem-solving and debugging skills.
7. Gained exposure to Git, Python libraries, and AI workflows.
Student Jul 2024 - Present
MLRIT, Hyderabad
Include achievements like:
CGPA: 9.19
Practicing Data Structures and Algorithms.
Learning Java, Python, SQL, HTML, CSS, JavaScript.
Building academic and personal projects.
Participated in multiple hackathons.
EDUCATION
B.Tech, Computer Science & Engineering 2024 - 2028
Marri Laxman Reddy Institute of Technology and Management
CGPA: 9.19/10
Senior Secondary (XII), Telangana State Board Of Intermediate Education 2024
Narayana Junior College
CGPA: 9.00/10
Secondary (X), CBSE 2022
Hindu Public School
CGPA: 8.00/10
SKILLS
•  Java •  Python •  C Programming
•  SQL •  MySQL •  VS Code
•  Google Colab •  Git •  GitHub
•  Pandas • NumPy •  Matplotlib
•  Seaborn •  Streamlit •  Data Analytics
•  Deep Learning •  Data Visualization •  Data Preprocessing
•  Problem Solving •  Analytical Thinking •  Presentation skills
ADDITIONAL DETAILS
- Secured 9.19 CGPA. - Participated in 3+ Hackathons. - Developed AI Job Readiness Prediction System. - Solved coding problems on
HackerRank and CodeChef. - Delivered technical seminar presentations. - Worked on AI/ML internship projects.
"""

documents = [Document(page_content=big_paragraph)]

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
)

docs = splitter.split_documents(documents)

vectorstore = FAISS.from_documents(docs, embeddings)

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

class RAGInput(BaseModel):
    input: str = Field(description="User Question")

runnable = (
    RunnableLambda(lambda x: x.input)
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

@app.get("/")
def home():
    return {
        "message": "RAG API is running successfully!",
        "docs": "/docs",
        "playground": "/rag/playground",
        "invoke": "/rag/invoke"
    }

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
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
    )
