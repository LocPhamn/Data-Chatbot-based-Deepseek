# RAG Chatbot with Deepseek-R1 Ollama

This project builds a Vietnamese Retrieval-Augmented Generation (RAG) chatbot for Tập đoàn Hòa Phát using LangChain, HuggingFace, FAISS, and Ollama. The chatbot can answer questions using both unstructured PDF data and structured Q&A samples from CSV.

## Features
- Ingests and indexes PDF documents and Q&A CSV samples into a FAISS vector database.
- Supports direct retrieval of Q&A samples or using them as context for LLM generation.
- Uses HuggingFace embedding models for semantic search.
- Integrates with Ollama for LLM inference (Deepseek-R1).
- Streamlit UI for interactive Q&A.

## Project Structure
- `vecto_db.py`: Build and update the FAISS vector database from PDF and CSV.
- `inference.py`: Inference pipeline, prompt construction, LLM integration, and Streamlit UI.
- `qabot.py`: (Optional) Additional chatbot logic.
- `data/hoaphatdata.csv`: Q&A sample data.
- `data/HoaPhatDoc.pdf`: Main PDF document.
- `directoryloader_preprocess/db_faiss/`: FAISS vector DB files.

## Setup Instructions
1. **Install dependencies**
   ```powershell
   pip install -r requirements.txt
   # or manually install: langchain, langchain-community, langchain-huggingface, sentence-transformers, streamlit, torch, etc.
   ```

2. **Download embedding model**
   - The default is `bkai-foundation-models/vietnamese-bi-encoder` (can be changed in code).
   - You may need to download the model manually if you have issues with meta tensor errors.

3. **Prepare data**
   - Place your PDF files in the `data/` folder.
   - Place your Q&A CSV (with columns `question`, `answer`) in `data/`.

4. **Build the vector database**
   ```powershell
   python vecto_db.py
   # This will create the FAISS vector DB from PDF and add Q&A samples as documents.
   ```

5. **Run the chatbot UI**
   ```powershell
   streamlit run inference.py
   # Open the provided local URL in your browser.
   ```

## Usage Notes
- The chatbot first tries to retrieve direct Q&A matches; otherwise, it uses RAG with LLM.
- You can update the vector DB with new Q&A samples using `add_csv_to_vectorstore` in `vecto_db.py`.
- Prompt engineering is handled in `inference.py` (see the `template` variable).

## Troubleshooting
- **Meta tensor error**: Download the embedding model locally and update the path in code.
- **Out of memory**: Use a lighter embedding model or reduce chunk size.
- **Streamlit watcher error**: The code disables the watcher by setting `STREAMLIT_WATCHER_TYPE=none`.

## Credits
- Built with LangChain, HuggingFace, FAISS, and Ollama.
- Embedding model: [bkai-foundation-models/vietnamese-bi-encoder](https://huggingface.co/bkai-foundation-models/vietnamese-bi-encoder)
- LLM: Deepseek-R1 via Ollama
