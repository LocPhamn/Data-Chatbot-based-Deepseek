from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_ollama import OllamaLLM
from transformers import AutoModelForCausalLM

from pprint import pprint
import streamlit as st
import os

# Tắt watcher của Streamlit để tránh lỗi với torch
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"

# Cấu hình đường dẫn model và vector DB
model_file = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
vector_db_path = "directoryloader_preprocess/db_faiss"


def load_llm_ollama():
    """
    Initialize Ollama LLM with DeepSeek model for Vietnamese QA.
    """
    return OllamaLLM(model="deepseek-r1:1.5b", base_url="http://localhost:11434", temperature=0.01)


def creat_prompt(template):
    """
    Create a PromptTemplate for the LLM with input variables 'question' and 'context'.
    """
    prompt = PromptTemplate(template=template, input_variables=["question", "context"])
    return prompt


def create_qa_chain(prompt, llm, db):
    """
    Build a RetrievalQA chain using the LLM, prompt, and vector database.
    """
    llm_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={"k": 3}, max_tokens_limit=2048),
        return_source_documents=False,
        chain_type_kwargs={'prompt': prompt}
    )
    return llm_chain


def read_vectors_db():
    """
    Load the FAISS vector database from disk using the specified embedding model.
    """
    embedding_model = HuggingFaceEmbeddings(model_name="bkai-foundation-models/vietnamese-bi-encoder")
    db = FAISS.load_local(vector_db_path, embedding_model, allow_dangerous_deserialization=True)
    return db

# Initialize main components

db = read_vectors_db()
llm = load_llm_ollama()

# Prompt for LLM

template = """
Bạn là trợ lý AI thông minh, chuyên trả lời các câu hỏi liên quan đến Tập đoàn Hòa Phát. Chỉ sử dụng thông tin sau đây trong vectodatabase để trả lời câu hỏi bằng tiếng Việt. Không được viết bằng tiếng Anh. và không sáng tạo nội dung""
Câu hỏi: {question}
Trả lời:
vectodatabase: {context}
"""

prompt = creat_prompt(template)
llm_chain = create_qa_chain(prompt, llm, db)

#Streamlit UI
if __name__ == '__main__':
    st.title("Hỏi đáp về Tập đoàn Hòa Phát")
    question = st.text_input("Nhập câu hỏi của bạn:")

    if st.button("Trả lời"):
        if question:
            with st.spinner("Đang lấy câu trả lời..."):
                response = llm_chain.invoke({"query": question})
            st.write(response['result'])
        else:
            st.write("Vui lòng nhập câu hỏi.")