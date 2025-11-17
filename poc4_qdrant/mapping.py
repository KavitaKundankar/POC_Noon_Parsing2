from typing import Dict, Any
from qdrant_client import QdrantClient, models
from qdrant_client.models import FieldCondition, MatchValue, Filter
from sentence_transformers import SentenceTransformer
from pathlib import Path
import json

# Qdrant + Embedding Model
client = QdrantClient("localhost", port=6333)
model = SentenceTransformer("multi-qa-MiniLM-L6-cos-v1")

COLLECTION = "noon_standard_params"

def map_single_key(extracted_key: str, tenant: str, vessel: str, threshold: float = 0.70):
    """Map one extracted key to standard key via vector search."""
    vector = model.encode(extracted_key).tolist()

    # ---------- 1) Vessel Level Match -------------
    vessel_level = client.search(
        collection_name=COLLECTION,
        query_vector=vector,
        limit=1,
        query_filter=Filter(
            must=[
                FieldCondition(key="tenant", match=MatchValue(value=tenant)),
                FieldCondition(key="vessel", match=MatchValue(value=vessel)),
            ]
        ),
    )

    if vessel_level and vessel_level[0].score >= threshold:
        payload = vessel_level[0].payload
        return payload["standard_key"], vessel_level[0].score, "vessel"

    # ---------- 2) Tenant Level Fallback ------------
    tenant_level = client.search(
        collection_name=COLLECTION,
        query_vector=vector,
        limit=1,
        query_filter=Filter(
            must=[
                FieldCondition(key="tenant", match=MatchValue(value=tenant))
            ]
        ),
    )

    if tenant_level and tenant_level[0].score >= threshold:
        payload = tenant_level[0].payload
        return payload["standard_key"], tenant_level[0].score, "tenant"

    # ---------- 3) Global fallback ------------
    global_match = client.search(
        collection_name=COLLECTION,
        query_vector=vector,
        limit=1
    )

    if global_match and global_match[0].score >= threshold:
        payload = global_match[0].payload
        return payload["standard_key"], global_match[0].score, "global"

    # Not mapped
    return None, None, None


def map_parsed_data(parsed_data: Dict[str, Any], tenant: str, vessel: str) -> Dict[str, Any]:
    """
    Takes parsed LLM output (dict) and maps all keys to standard parameters.
    """

    mapped = {}
    unmapped = {}

    print("\n🔄 Mapping Parsed Data to Standard Parameters...")

    for key, value in parsed_data.items():
        standard_key, score, level = map_single_key(key, tenant, vessel)

        if standard_key:
            print(f"✅ {key}  -->  {standard_key}   (match={score:.2f}, level={level})")
            mapped[standard_key] = value
        else:
            print(f"⚠️ No mapping found for key: {key}")
            unmapped[key] = value

    return {
        "mapped": mapped,
        "unmapped": unmapped,
        "vessel": vessel,
        "tenant": tenant
    }



tenant = "orion"
vessel = "KALIMANTAN EXPRESS"
parsed_file = Path("kalimantan_parsed.json")
with open(parsed_file, "r") as f:
        parsed_data = json.load(f)
result = map_parsed_data(parsed_data, tenant, vessel)

OUTPUT_FILE = Path("kalimantan_mapped6.json")
with open(OUTPUT_FILE, "w") as f:
    json.dump(result, f, indent=2)

print(f"✅ Output saved to {OUTPUT_FILE}")
