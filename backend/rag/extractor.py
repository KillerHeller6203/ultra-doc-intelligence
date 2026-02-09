import json
from langchain_community.llms import Ollama
from backend.ingestion.vector_store import VectorStore

class DataExtractor:
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
        self.llm = Ollama(
            model="llama3",
            temperature=0
        )

    def extract_fields(self):
        results = self.vector_store.search(
            "shipment details shipper consignee carrier rate weight",
            k=10
        )

        context = "\n---\n".join([res["text"] for res in results])

        schema = {
            "shipment_id": "Unique identifier for the shipment",
            "shipper": "Name of the sending party",
            "consignee": "Name of the receiving party",
            "pickup_datetime": "Date and time of pickup",
            "delivery_datetime": "Date and time of delivery",
            "equipment_type": "Type of equipment used",
            "mode": "Transportation mode",
            "rate": "Numeric rate value",
            "currency": "Currency code",
            "weight": "Numeric weight value",
            "carrier_name": "Name of the transportation company"
        }

        prompt = f"""
You are an information extraction engine.

Extract structured logistics data from the context below.
Return ONLY valid JSON.
If a field is missing, set it to null.

Schema:
{json.dumps(schema, indent=2)}

Context:
{context}

JSON:
"""

        try:
            response = self.llm.invoke(prompt)

            data = json.loads(response)

            return {field: data.get(field, None) for field in schema.keys()}

        except Exception:
            return {field: None for field in schema.keys()}
