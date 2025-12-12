# DataTalks-AI
# 📺 DataTalks-AI – RAG YouTuber Chatbot

A **Retrieval-Augmented Generation (RAG) chatbot** that answers questions using a YouTuber’s video transcripts.

Instead of scrolling through long videos, users can ask questions and receive **grounded, transcript-based answers**.
This project is built as a **learning-focused, end-to-end RAG system** covering vector search, LLMs, APIs, CI/CD, and cloud deployment.

---

## 🚀 Project Overview

The chatbot works as follows:

1. YouTube video transcripts are used as the knowledge source
2. Transcripts are chunked and embedded
3. Embeddings are stored in **LanceDB**
4. A user question triggers semantic retrieval
5. An LLM generates answers strictly from retrieved content
6. Responses are displayed in a simple chat interface

This ensures answers remain **accurate, grounded, and explainable**.

---

## ✨ Key Features

- Transcript-based question answering (RAG)
- Fast semantic search using a vector database
- API-first backend design with FastAPI
- Simple and clean chat UI built with Streamlit
- Serverless backend deployment on Azure Functions
- Automated CI/CD using GitHub Actions

---

## 🧠 Tech Stack

- Python
- LanceDB – Vector database
- PydanticAI – RAG orchestration
- Google Generative AI – Language model
- FastAPI – Backend API
- Streamlit – Frontend UI
- Azure Functions – Serverless backend
- Azure App Service – Frontend hosting
- GitHub Actions – CI/CD pipeline

---

## 📁 Repository Structure

```text
.
├── data/                # YouTube transcript files
├── lancedb/             # Vector database storage
├── src/                 # RAG logic, API, and Streamlit app
├── function_app.py      # Azure Functions entrypoint
├── .github/workflows/   # GitHub Actions workflows
└── requirements.txt
```

---

## 🖥️ Run Locally

### 1. Clone the repository and set up the environment

```bash
git clone https://github.com/Nosheen005/DataTalks-AI.git
cd DataTalks-AI

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

---

### 2. Environment Variables

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_key_here
```

---

### 3. Build the Knowledge Base (LanceDB)

Run your ingestion script or notebook that:

- Reads transcript files
- Splits transcripts into chunks
- Generates embeddings
- Stores embeddings in LanceDB

After this step, the chatbot is ready to answer questions.

---

### 4. Start the Application

#### Backend (FastAPI)

```bash
uvicorn src.api:app --reload --port 8000
```

#### Frontend (Streamlit)

```bash
streamlit run src/streamlit_app.py
```

---

## ☁️ Deployment on Azure (GitHub Actions)

The project is deployed in **two independent parts**, each managed via GitHub Actions.

### Backend – Azure Functions

- FastAPI runs as an Azure Function
- Entry point defined in `function_app.py`
- Automatically deployed on pushes to `main`
- Secrets managed via Azure App Settings

### Frontend – Streamlit on Azure App Service

- Streamlit app hosted on Azure App Service
- Automatically deployed using GitHub Actions
- Communicates with the backend via Azure Functions

---

## 🔄 CI/CD – GitHub Actions

- Every push to `main` triggers deployment
- Backend and frontend are deployed independently
- Secrets are stored securely in GitHub repository settings

---

## 🖼️ Screenshots

### Streamlit Chat Interface
![Streamlit UI](https://github.com/user-attachments/assets/f927a151-68bd-4686-bcf4-c4909f9b2769)

### Example Question and Answer
![Example Q&A](https://github.com/user-attachments/assets/a1bf4cea-68da-47b0-95f7-a501221ad923)

### Grounded Response from Transcript Data
![Grounded Response](https://github.com/user-attachments/assets/317060bc-c2e3-4b8b-9295-18fb6c3f19d7)

---

## 📝 Notes

- This project is built for **learning and experimentation**
- It provides a **solid foundation for real-world RAG applications**
