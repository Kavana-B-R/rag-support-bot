# 🚀 RAG-Based Customer Support Assistant (LangGraph + HITL)

## 📌 Overview

This project implements a **Retrieval-Augmented Generation (RAG)** based customer support assistant that answers user queries from a PDF knowledge base.

Unlike traditional chatbots, this system:

* Retrieves relevant information from documents
* Generates context-aware answers
* Uses a **graph-based workflow (LangGraph)**
* Supports **Human-in-the-Loop (HITL)** escalation for low-confidence queries

---

## 🎯 Features

* 📄 PDF-based knowledge ingestion
* 🔍 Semantic search using embeddings
* 🧠 Context-aware answer generation
* 🔁 Graph-based workflow using LangGraph
* ⚡ Conditional routing (answer vs escalation)
* 👨‍💻 Human-in-the-Loop fallback

---

## 🏗️ System Architecture

The system follows a RAG pipeline with decision-based workflow:

User Query → Retriever → LLM → Decision Node
↓
High Confidence → Answer
Low Confidence → HITL

---

## ⚙️ Tech Stack

* **Python**
* **LangChain**
* **LangGraph**
* **ChromaDB** (Vector Database)
* **Sentence Transformers** (Embeddings)
* **Ollama (LLaMA3)** – Local LLM

---

## 📂 Project Structure

```
rag-support-bot/
│
├── main.py              # Entry point
├── ingestion.py         # PDF loading & chunking
├── retriever.py         # Retrieval logic
├── graph.py             # LangGraph workflow
├── hitl.py              # Human escalation
├── requirements.txt     # Dependencies
├── sample.pdf           # Input knowledge base
```

---

## 🔄 How It Works

### 1. Document Processing

* Load PDF
* Split into chunks
* Convert to embeddings
* Store in ChromaDB

### 2. Query Processing

* User enters query
* Retrieve relevant chunks
* Generate answer using LLM

### 3. Decision Making

* If confidence is high → return answer
* If low confidence → escalate to human

---

## ▶️ Setup & Installation

### 1. Clone Repository

```bash
git clone https://github.com/Kavana-B-R/rag-support-bot.git
cd rag-support-bot
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Install Ollama (Local LLM)

Download from: https://ollama.com

Then run:

```bash
ollama pull llama3
```

---

## ▶️ Run the Project

```bash
python main.py
```

---

## 💬 Example Usage

**User Query:**

```
What is the refund policy?
```

**System Output:**

* Retrieves relevant content
* Generates answer

**If low confidence:**

```
⚠️ Escalating to human...
Enter human response:
```

---

## 🧠 Key Concepts Used

* Retrieval-Augmented Generation (RAG)
* Vector Databases
* Embeddings & Semantic Search
* Workflow Orchestration (LangGraph)
* Human-in-the-Loop Systems

---

## ⚠️ Limitations

* Basic confidence scoring
* Single document support
* No conversation memory

---

## 🚀 Future Improvements

* Multi-document support
* Chat memory
* Better confidence scoring
* Web UI (Streamlit)
* Deployment

---

## 📌 Author

Kavana B R

---

## ⭐ Acknowledgements

This project was built as part of a hands-on learning experience in **RAG systems and workflow-based AI design**.
