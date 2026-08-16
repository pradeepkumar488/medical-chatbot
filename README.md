# 🩺 Medical Chatbot (RAG-based)

An end-to-end medical question-answering chatbot built using Retrieval-Augmented Generation (RAG). The system retrieves relevant context from a medical reference document and generates accurate, concise answers using an LLM.

## 🚀 Features
- Answers medical questions using real reference document context (not hallucinated)
- RAG pipeline: PDF ingestion → chunking → embeddings → vector search → LLM response
- Free, fast inference using Groq's LLaMA 3.1 model
- Simple, responsive chat interface built with Flask

## 🛠️ Tech Stack
- **Backend:** Flask
- **LLM:** Groq (LLaMA 3.1 8B Instant)
- **Orchestration:** LangChain
- **Vector Database:** Pinecone
- **Embeddings:** HuggingFace Sentence Transformers (`all-MiniLM-L6-v2`)
- **Frontend:** HTML, CSS, JavaScript (jQuery, Bootstrap)

## 📂 Project Structure
medical-chatbot/
├── src/
│ ├── helper.py # PDF loading, chunking, embeddings
│ └── prompt.py # System prompt for the LLM
├── static/
│ └── style.css # Chat UI styling
├── templates/
│ └── chat.html # Chat UI
├── data/ # Medical reference PDF(s)
├── app.py # Flask app (chatbot backend)
├── store_index.py # One-time script to build the vector index
└── requirements.txt
## ⚙️ How It Works
1. A medical reference PDF is loaded and split into text chunks
2. Each chunk is converted into a vector embedding
3. Embeddings are stored in a Pinecone vector database
4. On a user query, the most relevant chunks are retrieved
5. The retrieved context + user question are passed to the Groq LLM
6. The LLM generates a concise, context-grounded answer

## 🔧 Setup & Installation

1. **Clone the repository**
```bash
git clone https://github.com/pradeepkumar488/medical-chatbot.git
cd medical-chatbot
```

2. **Create a virtual environment**
```bash
python -m venv medibot
medibot\Scripts\activate     # Windows
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Add your API keys**
Create a `.env` file in the root directory:
5. **Add your medical PDF**
Place a medical reference PDF inside the `data/` folder.

6. **Build the vector index** (run once)
```bash
python store_index.py
```

7. **Run the app**
```bash
python app.py
```

8. **Open in browser**
http://localhost:8080
## 📌 Notes
- This chatbot is for educational/informational purposes only and is **not a substitute for professional medical advice**.
- Answers are limited to the context available in the ingested PDF document.

## 👤 Author
**Pradeepa Kumara**
GitHub: [@pradeepkumar488](https://github.com/pradeepkumar488)
