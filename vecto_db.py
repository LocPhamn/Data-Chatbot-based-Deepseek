from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
import re

# Khai bao bien
pdf_data_path = "data"
vector_db_path = "vectorstores_v2/db_faiss"

def preprocess_text(text):
    # Loại bỏ header/footer phổ biến (ví dụ: tên công ty, số trang)
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        line = line.strip()
        # Loại bỏ dòng trống, dòng chỉ có số (số trang), dòng rất ngắn
        if not line or line.isdigit() or len(line) < 4:
            continue
        # Loại bỏ các header/footer phổ biến (tùy chỉnh theo file PDF)
        if re.search(r'Hòa Phát|Báo cáo thường niên|Page|Trang|Công ty cổ phần', line, re.IGNORECASE):
            continue
        cleaned_lines.append(line)
    # Gộp lại thành đoạn văn
    text = ' '.join(cleaned_lines)
    # Loại bỏ nhiều khoảng trắng liên tiếp
    text = re.sub(r'\s+', ' ', text)
    # Loại bỏ ký tự đặc biệt không cần thiết (tùy chỉnh thêm nếu cần)
    text = re.sub(r'[^\x00-\x7FÀ-ỹ.,;:?!%()\\/-]', '', text)
    return text


def create_db_from_files():
    # Khai bao loader de quet toan bo thu muc dataa
    loader = DirectoryLoader(pdf_data_path, glob="*.pdf", loader_cls = PyPDFLoader)
    documents = loader.load()
    for doc in documents:
        doc.page_content = preprocess_text(doc.page_content)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=300)
    chunks = text_splitter.split_documents(documents)

    # Embeding
    embedding_model = HuggingFaceEmbeddings(model_name="bkai-foundation-models/vietnamese-bi-encoder")
    db = FAISS.from_documents(chunks, embedding_model)
    db.save_local(vector_db_path)
    return db

def print_loaded_text():
    loader = DirectoryLoader(pdf_data_path, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    for doc in documents:
        doc.page_content = preprocess_text(doc.page_content)
    for i, doc in enumerate(documents):
        print(f"--- Document {i+1} ---")
        print(len(doc.page_content[:]))  # In 1000 ký tự đầu tiên của mỗi trang/document
        print("\n")


# Gọi hàm để in ra text đã load từ PDF
# print_loaded_text()

# Tạo vector DB từ file
create_db_from_files()