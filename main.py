import math
import os
import json
import time
from typing import List, Dict, Optional
import torch
import streamlit as st
import concurrent.futures
from langchain_community.llms import Ollama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

# Basic config
PDF_FOLDER = "./pdf_folder"
VECTOR_STORE_DIR = "./vector_store"
OLLAMA_BASE_URL = "http://localhost:11434"
MODEL_NAME = "phi4"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 250
CHUNK_OVERLAP = 50
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class PDFProcessor:
    def __init__(
        self,
        pdf_folder: str = PDF_FOLDER,
        vector_store_dir: str = VECTOR_STORE_DIR,
        model_name: str = MODEL_NAME,
        embedding_model: str = EMBEDDING_MODEL,
        ollama_base_url: str = OLLAMA_BASE_URL,
        device: str = DEVICE
    ):
        # Setup directories
        os.makedirs(pdf_folder, exist_ok=True)
        os.makedirs(vector_store_dir, exist_ok=True)
        self.pdf_folder = pdf_folder
        self.device = device

        # Initialize components
        self.llm = Ollama(model=model_name, base_url=ollama_base_url)
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={"device": self.device},
            encode_kwargs={"normalize_embeddings": True}
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        self.vector_store = Chroma(
            collection_name="pdf_collection",
            persist_directory=vector_store_dir,
            embedding_function=self.embeddings
        )

    def generate_summaries(self, force: bool = False) -> List[Dict]:
        summary_path = os.path.join(self.pdf_folder, "summaries.json")
        summaries = {}

        if os.path.exists(summary_path) and not force:
            with open(summary_path, 'r') as f:
                summaries = json.load(f)

        updated_summaries = {}

        for filename in os.listdir(self.pdf_folder):
            if not filename.lower().endswith(".pdf"):
                continue

            if filename in summaries and not force:
                updated_summaries[filename] = summaries[filename]
                continue

            try:
                file_path = os.path.join(self.pdf_folder, filename)
                loader = PyPDFLoader(file_path)
                pages = loader.load()
                first_page = pages[0].page_content if pages else ""
                first_page_lines = first_page.split('\n')[:1000]
                first_page = '\n'.join(first_page_lines)
                
                prompt = f"Give an indepth brief outline about the the following content:\n\n{first_page}"
                summary = self.llm.invoke(prompt).strip()
                updated_summaries[filename] = summary
            except Exception:
                continue

        with open(summary_path, 'w') as f:
            json.dump(updated_summaries, f, indent=2)

        return [{"filename": fname, "summary": summary} for fname, summary in updated_summaries.items()]

    def get_pdf_list(self) -> List[Dict]:
        try:
            summary_path = os.path.join(self.pdf_folder, "summaries.json")
            if not os.path.exists(summary_path):
                return self.generate_summaries()

            with open(summary_path, 'r') as f:
                summaries = json.load(f)

            return [{"filename": fname, "summary": summary} for fname, summary in summaries.items()]
        except Exception:
            return []

    def select_relevant_files(self, query: str, pdf_list: List[Dict]) -> List[str]:
        if not pdf_list:
            return []
        
        try:
            query_embedding = self.embeddings.embed_query(query)
            summary_texts = [f"{pdf['filename']}: {pdf['summary']}" for pdf in pdf_list]
            summary_embeddings = self.embeddings.embed_documents(summary_texts)
            
            similarities = []
            for i, emb in enumerate(summary_embeddings):
                similarity = self._cosine_similarity(query_embedding, emb)
                similarities.append((pdf_list[i]['filename'], similarity))
            
            similarities.sort(key=lambda x: x[1], reverse=True)
            return [item[0] for item in similarities[:3]]
            
        except Exception:
            file_descriptions = "\n".join([
                f"File: {pdf['filename']}\nSummary: {pdf['summary']}\n"
                for pdf in pdf_list
            ])
            
            selection_prompt = f"""Given the following query and list of PDF files with their summaries, 
            select the top 3 most relevant files that would best answer the query.
            Return ONLY the exact filenames in a comma-separated list without explanations or additional text.

            Query: {query}

            Available files:
            {file_descriptions}

            Top 3 relevant files (exact filenames only):"""

            response = self.llm.invoke(selection_prompt)
            raw_files = response.replace('\n', ',').split(',')
            selected_files = []
            available_filenames = [pdf['filename'] for pdf in pdf_list]
            
            for file in raw_files:
                clean_file = file.strip()
                if clean_file in available_filenames and clean_file not in selected_files:
                    selected_files.append(clean_file)
            
            if len(selected_files) < 3:
                for file in raw_files:
                    clean_file = file.strip()
                    if clean_file not in selected_files:
                        for filename in available_filenames:
                            if (clean_file.lower() in filename.lower() or 
                                filename.lower() in clean_file.lower()) and filename not in selected_files:
                                selected_files.append(filename)
                                break
                    if len(selected_files) >= 3:
                        break
            
            if len(selected_files) < 3 and len(available_filenames) > 0:
                query_keywords = set(query.lower().split())
                content_relevance = []
                
                for pdf in pdf_list:
                    if pdf['filename'] not in selected_files:
                        summary_text = pdf['summary'].lower()
                        matches = sum(1 for keyword in query_keywords if keyword in summary_text)
                        if matches > 0:
                            content_relevance.append((pdf['filename'], matches))
                
                content_relevance.sort(key=lambda x: x[1], reverse=True)
                for filename, _ in content_relevance:
                    if filename not in selected_files:
                        selected_files.append(filename)
                    if len(selected_files) >= 3:
                        break
            
            if len(selected_files) < 3 and len(available_filenames) > 0:
                for filename in available_filenames:
                    if filename not in selected_files:
                        selected_files.append(filename)
                    if len(selected_files) >= 3:
                        break
            
            return selected_files
        except Exception:
            return available_filenames[:3] if available_filenames else []

    # Use an algorithm to calculate the cosine similarity between two vectors so we can find relevant files. Last resort if the LLM fails to select relevant files.
    def _cosine_similarity(self, vector_a, vector_b):
        dot_product = sum(a * b for a, b in zip(vector_a, vector_b))
        magnitude_a = math.sqrt(sum(a * a for a in vector_a))
        magnitude_b = math.sqrt(sum(b * b for b in vector_b))
        if magnitude_a == 0 or magnitude_b == 0:
            return 0
        return dot_product / (magnitude_a * magnitude_b)

    def process_pdf(self, file_path: str) -> bool:
        try:
            if not os.path.exists(file_path):
                return False

            loader = PyPDFLoader(file_path)
            docs = loader.load_and_split(text_splitter=self.text_splitter)
            
            if not docs:
                return False
            
            batch_size = 64
            for i in range(0, len(docs), batch_size):
                batch = docs[i:i + batch_size]
                for doc in batch:
                    doc.metadata["source"] = os.path.basename(file_path)
                
                self.vector_store.add_documents(
                    documents=batch,
                    ids=[f"{os.path.basename(file_path)}chunk{i+j}" for j in range(len(batch))]
                )
            
            self.vector_store.persist()
            return True
        except Exception:
            return False

    def process_selected_files(self, selected_files: List[str]) -> bool:
        if not selected_files:
            return False
                
        def process_file(filename: str) -> bool:
            file_path = os.path.join(self.pdf_folder, filename)
            return self.process_pdf(file_path)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(process_file, filename) for filename in selected_files]
            results = [future.result() for future in futures]
        
        return any(results)

    def create_retriever(self, selected_files: Optional[List[str]] = None) -> Optional[RetrievalQA]:
        try:
            search_kwargs = {"k": 5}
            
            if selected_files:
                search_kwargs["filter"] = {"source": {"$in": selected_files}}

            retriever = self.vector_store.as_retriever(search_kwargs=search_kwargs)
            
            qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=True
            )
            
            return qa_chain
        except Exception:
            return None

    def query_documents(self, query: str, selected_files: Optional[List[str]] = None) -> Dict:
        try:
            qa_chain = self.create_retriever(selected_files)
            if not qa_chain:
                return {"error": "Failed to create retriever"}

            result = qa_chain.invoke({"query": query})
            
            sources = []
            if "source_documents" in result:
                for doc in result["source_documents"]:
                    source = doc.metadata.get("source", "Unknown")
                    if source and {"source": source} not in sources:
                        sources.append({"source": source})

            return {
                "answer": result.get("result", "No answer could be generated."),
                "sources": sources
            }
        except Exception as e:
            return {"error": str(e)}

def upload_pdfs():
    uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)
    
    if uploaded_files:
        with st.spinner('Uploading PDF files...'):
            for uploaded_file in uploaded_files:
                file_path = os.path.join(PDF_FOLDER, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                st.success(f"Uploaded: {uploaded_file.name}")
        
        st.session_state.pdf_processor.generate_summaries()
        st.session_state.pdf_list = st.session_state.pdf_processor.get_pdf_list()
        st.experimental_rerun()

def show_pdfs():
    pdf_list = st.session_state.pdf_list
    
    if not pdf_list:
        st.warning(f"No PDF files found in {os.path.abspath(PDF_FOLDER)}. Please upload some PDFs.")
        return
    
    st.subheader(f"Available PDFs ({len(pdf_list)})")
    
    with st.expander("View PDF summaries"):
        for pdf in pdf_list:
            st.markdown(f"**{pdf['filename']}**")
            st.markdown(f"{pdf['summary']}")
            st.divider()

def handle_query():
    query = st.session_state.get("user_query", "")
    
    if not query:
        return
    
    pdf_list = st.session_state.pdf_list
    
    if not pdf_list:
        st.error("No PDFs available. Please upload some PDFs first.")
        return
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    st.session_state.chat_history.append({"role": "user", "content": query})
    
    with st.spinner('Searching through PDFs...'):
        selected_files = st.session_state.pdf_processor.select_relevant_files(query, pdf_list)
        
        if not selected_files:
            selected_files = [item['filename'] for item in pdf_list][:3]
        
        st.session_state.pdf_processor.process_selected_files(selected_files)
        result = st.session_state.pdf_processor.query_documents(query, selected_files)
        
        if "error" in result:
            answer = f"An error occurred: {result['error']}"
            sources = []
        else:
            answer = result["answer"]
            sources = result["sources"]
        
        st.session_state.chat_history.append({
            "role": "assistant", 
            "content": answer,
            "sources": sources,
            "selected_files": selected_files
        })
    
    st.session_state.user_query = ""

def display_chat():
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            st.chat_message("user").write(message["content"])
        else:
            with st.chat_message("assistant"):
                st.write(message["content"])
                
                if "sources" in message and message["sources"]:
                    with st.expander("Sources"):
                        for source in message["sources"]:
                            st.write(f"- {source['source']}")
                
                if "selected_files" in message and message["selected_files"]:
                    with st.expander("Files used"):
                        for file in message["selected_files"]:
                            st.write(f"- {file}")

def initialize_session():
    if "initialized" not in st.session_state:
        try:
            os.makedirs(PDF_FOLDER, exist_ok=True)
            os.makedirs(VECTOR_STORE_DIR, exist_ok=True)
            
            st.session_state.pdf_processor = PDFProcessor()
            st.session_state.pdf_list = st.session_state.pdf_processor.get_pdf_list()
            
            if "chat_history" not in st.session_state:
                st.session_state.chat_history = []
            
            st.session_state.initialized = True
            
        except Exception as e:
            st.error(f"Error initializing application: {str(e)}")

def main():
    st.set_page_config(
        page_title="PDF Question Answering App",
        page_icon="📚",
        layout="wide"
    )
    
    st.title("📚 PDF Question Answering App")
    
    initialize_session()
    
    with st.sidebar:
        st.header("PDF Management")
        upload_pdfs()
        st.divider()
        show_pdfs()
        
        with st.expander("Technical Information"):
            st.write(f"Using device: {DEVICE}")
            st.write(f"LLM: {MODEL_NAME}")
            st.write(f"Embedding model: {EMBEDDING_MODEL}")
            st.write(f"PDF folder: {os.path.abspath(PDF_FOLDER)}")
            st.write(f"Vector store: {os.path.abspath(VECTOR_STORE_DIR)}")
    
    display_chat()
    
    user_query = st.chat_input("Ask a question about your PDFs...")
    if user_query:
        st.session_state.user_query = user_query
        handle_query()
        st.experimental_rerun()

if __name__ == "__main__":
    main()