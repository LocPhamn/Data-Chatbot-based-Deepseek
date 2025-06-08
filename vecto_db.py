import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_unstructured import UnstructuredLoader


# Khai bao bien
pdf_data_path = "data"
file_paths = [
    "./data/HoaPhatDoc.pdf",
]

vector_db_path = "directoryloader_preprocess/db_faiss"


def load_qa_csv_to_documents(csv_path):
    """
    Load Q&A samples from a CSV file and convert each row to a Document object.
    """
    df = pd.read_csv(csv_path)
    documents = []

    for _, row in df.iterrows():
        content = f"Câu hỏi: {row['question']}\nTrả lời: {row['answer']}"
        documents.append(Document(page_content=content))

    return documents


def create_db_from_files(qa_samples_context=None):
    """
    Create a FAISS vector database from PDF files and optionally add a Q&A context document.
    """
    loader = DirectoryLoader(pdf_data_path, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()

    # If Q&A context is provided, add as a single document
    if qa_samples_context:
        qa_doc = Document(page_content=qa_samples_context, metadata={"source": "qa_samples"})
        documents.append(qa_doc)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)

    # Embedding
    embedding_model = HuggingFaceEmbeddings(model_name="bkai-foundation-models/vietnamese-bi-encoder")
    db = FAISS.from_documents(chunks, embedding_model)
    db.save_local(vector_db_path)
    return db


def add_csv_to_vectorstore(csv_path, vectorstore, embedding_model=None):
    """
    Add Q&A samples from a CSV file as individual documents to an existing vectorstore.
    If embedding_model is not provided, a default one will be created.
    """
    if embedding_model is None:
        embedding_model = HuggingFaceEmbeddings(model_name="bkai-foundation-models/vietnamese-bi-encoder")
    documents = load_qa_csv_to_documents(csv_path)
    vectorstore.add_documents(documents)
    vectorstore.save_local(vector_db_path)
    print(f"Đã thêm {len(documents)} Q&A mẫu vào vectorstore và lưu lại.")


if __name__ == '__main__':
    # Tạo vector DB từ PDF
    db = create_db_from_files()
    # Thêm các Q&A mẫu từ CSV vào vector DB dưới dạng nhiều documents
    add_csv_to_vectorstore("data/hoaphatdata.csv", db)