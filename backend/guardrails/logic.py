import numpy as np

class GuardrailLogic:
    def __init__(self, similarity_threshold=0.35, confidence_threshold=0.35):
        # similarity_threshold: Minimum similarity for the best chunk (1 / (1 + distance))
        # confidence_threshold: Minimum aggregate confidence score
        self.similarity_threshold = similarity_threshold
        self.confidence_threshold = confidence_threshold

    def _distance_to_similarity(self, distance: float) -> float:
        """Convert L2 distance to a 0-1 similarity score."""
        return 1 / (1 + distance)

    def calculate_confidence(self, search_results: list[dict]) -> dict:
        """
        Calculate a heuristic-based confidence score (0-1).
        Formula: 0.6 * max_similarity + 0.4 * avg_similarity
        """
        if not search_results:
            return {"confidence_score": 0.0, "max_similarity": 0.0, "avg_similarity": 0.0}
        
        similarities = [self._distance_to_similarity(res["score"]) for res in search_results]
        max_sim = max(similarities)
        avg_sim = sum(similarities) / len(similarities)
        
        confidence = (0.6 * max_sim) + (0.4 * avg_sim)
        confidence = min(1.0, confidence)
        
        return {
            "confidence_score": round(confidence, 2),
            "max_similarity": round(max_sim, 2),
            "avg_similarity": round(avg_sim, 2)
        }

    def validate_request(self, search_results: list[dict]) -> tuple[bool, str, dict]:
        """
        Enforce strict hallucination guardrails.
        Returns: (is_valid, reason, stats)
        """
        stats = self.calculate_confidence(search_results)
        conf = stats["confidence_score"]
        max_sim = stats["max_similarity"]

        if max_sim < self.similarity_threshold:
            return False, "Not found in document (Low retrieval similarity).", stats
        
        if conf < self.confidence_threshold:
            return False, "Refused answer (Low aggregate confidence).", stats
            
        return True, "", stats
