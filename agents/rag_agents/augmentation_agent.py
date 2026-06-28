class AugmentationAgent:
    def run(self,state):
        chunks=state["retrieved_chunks"]
        augmented_context="\n\n".join(chunks)
        print(augmented_context)

        return {"augmented_context":augmented_context}
    