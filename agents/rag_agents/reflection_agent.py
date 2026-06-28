class ReflectionAgent:

    def _score_overlap(self, source_text, target_text):
        """
        Simple overlap-based scoring.
        Returns score between 0 and 1.
        """

        source_words = set(source_text.lower().split())
        target_words = set(target_text.lower().split())

        if len(source_words) == 0:
            return 0.0

        overlap = len(source_words.intersection(target_words))

        return round(overlap / len(source_words), 2)

    def run(self, state):

        query = state["query"]
        answer = state["generated_answer"]
        context = state["augmented_context"]

        if not answer.strip():
            return {
                "groundedness_score": 0.0,
                "answer_relevance_score": 0.0,
                "context_relevance_score": 0.0,
                "safety_label": "Unsafe",
                "grounded": False,
                "final_answer": "Unable to answer safely"
            }

        groundedness_score = self._score_overlap(
            answer,
            context
        )

        answer_relevance_score = self._score_overlap(
            query,
            answer
        )

        context_relevance_score = self._score_overlap(
            query,
            context
        )

        avg_score = (
            groundedness_score +
            answer_relevance_score +
            context_relevance_score
        ) / 3

        if avg_score >= 0.7:
            safety_label = "Safe"
        elif avg_score >= 0.4:
            safety_label = "Needs Review"
        else:
            safety_label = "Unsafe"

        grounded = groundedness_score >= 0.7

        return {
            "groundedness_score": groundedness_score,
            "answer_relevance_score": answer_relevance_score,
            "context_relevance_score": context_relevance_score,
            "safety_label": safety_label,
            "grounded": grounded,
            "final_answer": (
                answer
                if safety_label != "Unsafe"
                else "Unable to answer safely"
            )
        }