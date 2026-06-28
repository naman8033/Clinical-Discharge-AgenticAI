from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
load_dotenv()
api_key=os.getenv("FOUNDATION_MODEL_API_KEY")
base_url=os.getenv("LLM_BASE_URL")

class IndexingAgent:
    def __init__(self):
        self.splitter=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=100)
        self.embeddings=OpenAIEmbeddings(model="text-embedding-3-small",api_key=api_key,base_url=base_url)

    def run(self,report_text):
        chunks=self.splitter.split_text(report_text)
        print("chunk created:")
        for i, chunk in enumerate(chunks,start=1):
            print(f"chunk {i}")
            print(chunk)
            print("-"*15)

        docs=[Document(page_content=chunk) for chunk in chunks]
        vector_db=FAISS.from_documents(docs,self.embeddings)
        vector_db.save_local("faiss_index")

        return vector_db