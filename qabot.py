import pandas as pd
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
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"

# Cau hinh
model_file = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
vector_db_path = "directoryloader_preprocess/db_faiss"


def load_llm_ollama():
    return OllamaLLM(model="deepseek-r1:1.5b", base_url="http://localhost:11434", temperature=0.01,)

# Tao prompt template
def creat_prompt(template):
    prompt = PromptTemplate(template=template, input_variables=["question","context",])
    return prompt


# Tao simple chain
def create_qa_chain(prompt, llm, db):
    llm_chain = RetrievalQA.from_chain_type(
        llm = llm,
        chain_type= "stuff",
        retriever = db.as_retriever(search_kwargs = {"k":3}, max_tokens_limit=2048),
        return_source_documents = False,
        chain_type_kwargs= {'prompt': prompt}

    )
    return llm_chain

# Read tu VectorDB
def read_vectors_db():
    embedding_model = HuggingFaceEmbeddings(model_name="bkai-foundation-models/vietnamese-bi-encoder")
    db = FAISS.load_local(vector_db_path, embedding_model,allow_dangerous_deserialization=True)
    return db


db = read_vectors_db()
llm = load_llm_ollama()

template = """
Bạn là trợ lý AI thông minh, chuyên trả lời các câu hỏi liên quan đến Tập đoàn Hòa Phát. Chỉ sử dụng thông tin sau đây trong vectodatabase để trả lời câu hỏi bằng tiếng Việt. Không được viết bằng tiếng Anh. và không sáng tạo nội dung""
Câu hỏi: {question}
Trả lời:
vectodatabase: {context}
"""

prompt = creat_prompt(template)
llm_chain  =create_qa_chain(prompt, llm, db)

# question = "Trong năm 2023, Hòa Phát đóng góp bao nhiêu tiền vào ngân sách Nhà nước"
# response = llm_chain.invoke({"query": question})
# pprint(response['result'])

if __name__ == '__main__':
    st.title("Hỏi đáp về Tập đoàn Hòa Phát")
    question = st.text_input("Nhập câu hỏi của bạn:")

    if st.button("Trả lời"):
        if question:
            response = llm_chain.invoke({"query": question})
            st.write(response['result'])
        else:
            st.write("Vui lòng nhập câu hỏi.")
