from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from agents.llm import gateway_kwargs, get_embeddings
load_dotenv()

INDEX_DIR = "faiss_index"

class IndexingAgent:
    def __init__(self):
        self.splitter=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=100)
        self.embeddings = get_embeddings()

    def run(self,report_text,patient_id=None):
        print(f"[IndexingAgent] received text of length {len(report_text)} for patient_id={patient_id}")
        chunks=self.splitter.split_text(report_text)
        print(f"[IndexingAgent] split into {len(chunks)} chunks")
        for i, chunk in enumerate(chunks,start=1):
            print(f"chunk {i}")
            print(chunk)
            print("-"*15)

        docs=[Document(page_content=chunk,metadata={"patient_id":patient_id}) for chunk in chunks]

        index_file = os.path.join(INDEX_DIR, "index.faiss")
        vector_db = None

        if os.path.exists(index_file):
            print(f"[IndexingAgent] existing index found at ./{INDEX_DIR}, loading and checking dimension...")
            try:
                loaded_db = FAISS.load_local(INDEX_DIR, self.embeddings, allow_dangerous_deserialization=True)
                sample_dim = len(self.embeddings.embed_query("test"))
                if loaded_db.index.d == sample_dim:
                    loaded_db.add_documents(docs)
                    vector_db = loaded_db
                    print(f"[IndexingAgent] successfully appended docs to existing index.")
                else:
                    print(f"[IndexingAgent] Dimension mismatch: index has {loaded_db.index.d}, current embeddings produce {sample_dim}. Recreating index...")
            except Exception as e:
                print(f"[IndexingAgent] Could not append to existing index ({e}), recreating new index...")

        if vector_db is None:
            print(f"[IndexingAgent] creating new index at ./{INDEX_DIR}...")
            vector_db = FAISS.from_documents(docs, self.embeddings)

        vector_db.save_local(INDEX_DIR)
        print(f"[IndexingAgent] saved FAISS index to ./{INDEX_DIR} (now contains data for patient_id={patient_id})")

        return vector_db