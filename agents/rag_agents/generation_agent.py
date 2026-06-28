from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
load_dotenv()
api_key=os.getenv("FOUNDATION_MODEL_API_KEY")
base_url=os.getenv("LLM_BASE_URL")

llm=ChatOpenAI(
    model="gpt-5.4-mini",
    temperature=0.7,
    api_key=api_key,
    base_url=base_url
)

class GenerationAgent:
    def run(self,state):
        query=state["query"]
        context=state["augmented_context"]
        prompt=f"""
        You are a clinical QA assistant.
        Answer only from the provided context.
        If the answer is not present in the context, respond with:
        "I could not find the answer in the provided records"
        Context:
        {context}
        Question:
        {query}
        """

        print("----------------------------------------")
        print(query)
        print(context)

        print("-----------------------------------------------------------------")

        response=llm.invoke(prompt)
        print(response)
        return {"generated_answer":response.content}