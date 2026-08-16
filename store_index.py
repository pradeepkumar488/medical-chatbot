from src.helper import load_pdf_file, text_split, download_hugging_face_embeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import os

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

print("Loading PDF...")
extracted_data = load_pdf_file(data='data/')

print("Splitting text into chunks...")
text_chunks = text_split(extracted_data)
print(f"Total chunks: {len(text_chunks)}")

print("Loading embedding model...")
embeddings = download_hugging_face_embeddings()

pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "medical-chatbot"

# Create index only if it doesn't already exist
if index_name not in [i["name"] for i in pc.list_indexes()]:
    print("Creating Pinecone index...")
    pc.create_index(
        name=index_name,
        dimension=384,   # matches all-MiniLM-L6-v2 output size
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
else:
    print("Index already exists, skipping creation.")

print("Uploading chunks to Pinecone... (this may take a while)")
docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    index_name=index_name,
    embedding=embeddings,
)

print("Done! Your medical data is now searchable in Pinecone.")