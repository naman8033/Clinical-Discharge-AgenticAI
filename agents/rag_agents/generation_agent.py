from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from agents.llm import gateway_kwargs, get_default_model
load_dotenv()

llm=ChatOpenAI(
    model=get_default_model(),
    temperature=0.7,
    **gateway_kwargs(),
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