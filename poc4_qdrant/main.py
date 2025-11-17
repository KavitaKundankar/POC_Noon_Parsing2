import os
import json
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
from qdrant_client.models import PointStruct

# --- CONFIG ---
COLLECTION = "noon_standard_params"
JSON_FILE = "standard_parameter_input/orion_standardparameter_description_list.json"

client = QdrantClient("localhost", port=6333)
model = SentenceTransformer("multi-qa-MiniLM-L6-cos-v1")

client.recreate_collection(
    collection_name=COLLECTION,
    vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE),
)

points = []
point_id = 1

with open(JSON_FILE, "r") as f:
    data = json.load(f)


for item in data:
    text_to_embed = f"{item['name']} - {item['description']}"
    vector = model.encode(text_to_embed).tolist()

    payload = {
        "standard_key": item["name"],
        "description": item["description"],
    }

    points.append(PointStruct(id=point_id, vector=vector, payload=payload))
    point_id += 1

client.upsert(collection_name=COLLECTION, points=points)
print(f"Stored {len(points)} vectors into Qdrant!")
