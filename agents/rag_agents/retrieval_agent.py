class RetrievalAgent:

    def run(self,vector_db,query):
        docs=vector_db.similarity_search(query,k=3)
        retrieved_chunks=[doc.page_content for doc in docs]
        print(retrieved_chunks)
        return {"retrieved_chunks":retrieved_chunks}
    