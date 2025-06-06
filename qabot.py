from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

from pprint import pprint

# Cau hinh
model_file = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
vector_db_path = "vectorstores_v2/db_faiss"

# Load LLM
def load_llm(model_file):
    # llm = CTransformers(
    #     model=model_file,
    #     model_type="llama",
    #     max_new_tokens=1024,
    #     temperature=0.01
    # )
    tokenizer = AutoTokenizer.from_pretrained(model_file)
    model = AutoModelForCausalLM.from_pretrained(model_file)
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=512, device= -1, temperature=0.001)
    llm = HuggingFacePipeline(pipeline=pipe)
    return llm

# Tao prompt template
def creat_prompt(template):
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])
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
    # Embeding
    # embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    embedding_model = HuggingFaceEmbeddings(model_name="bkai-foundation-models/vietnamese-bi-encoder")

    db = FAISS.load_local(vector_db_path, embedding_model,allow_dangerous_deserialization=True )
    return db


# Bat dau thu nghiem
db = read_vectors_db()
llm = load_llm(model_file)

#Tao Prompt
# template = """<|im_start|>system\nSử dụng thông tin sau đây để trả lời câu hỏi. Nếu bạn không biết câu trả lời, hãy nói không biết, đừng cố tạo ra câu trả lời\n
#     {context}<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant"""

template = """
Bạn là trợ lý AI thông minh, chuyên trả lời các câu hỏi liên quan đến Tập đoàn Hòa Phát. Chỉ sử dụng thông tin sau đây (DB) để trả lời câu hỏi. 
Nếu không tìm thấy câu trả lời trong DB, hãy trả lời: "Tôi không biết."
Câu hỏi: {question}
DB: {context}
Trả lời:
"""

prompt = creat_prompt(template)

llm_chain  =create_qa_chain(prompt, llm, db)

# Chay cai chain
question = "Tầm nhìn của tập đoàn hòa phát là gì "
response = llm_chain.invoke({"query": question})
print("Context truyền vào:")
print(response.get("context", ""))
print("Kết quả:")
print(response.get("result", ""))