from langchain_community.llms import Ollama
from backend.ingestion.vector_store import VectorStore
from backend.guardrails.logic import GuardrailLogic


class RAGEngine:
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
        self.guardrails = GuardrailLogic()
        self.llm = Ollama(
            model="llama3",
            temperature=0
        )

    def ask(self, question: str):
        results = self.vector_store.search(question, k=4)
        is_valid, reason, stats = self.guardrails.validate_request(results)
        if not is_valid:
            return {
                "answer": reason,
                "sources": [],
                "confidence_score": stats["confidence_score"]
            }
        context = "\n---\n".join([res["text"] for res in results])

        prompt = f"""
Answer strictly using the information present in the document excerpts below.
If the requested information is not present, respond with:
"Information not available in the document."

Document Excerpts:
{context}

Query: {question}
Response:"""

        answer = self.llm.invoke(prompt)

        return {
            "answer": answer,
            "sources": [res["text"] for res in results],
            "confidence_score": stats["confidence_score"]
        }
