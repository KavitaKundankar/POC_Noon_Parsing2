import os
import json
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

DB_DIR = "standard_param_db_misuga_description"

def load_param_db():
    """Load Chroma DB with metadata {name, description}."""
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectordb = Chroma(
        persist_directory=DB_DIR,
        embedding_function=embeddings
    )

    return vectordb


def rag_map_key(llm, extracted_key, candidates):
    """LLM chooses the best matching standard parameter key."""

    prompt = f"""
You are an expert maritime data mapper.

Extracted key:
"{extracted_key}"

Candidate standard parameters (from vector DB):
{json.dumps(candidates, indent=2)}

Rules:
1. Return ONLY the best matching parameter name.
2. Do NOT return description or explanation.
3. For FO/DO/GO keys, map based on sulphur class (MAX 0.5, OVER 0.5, etc.).
4. For RPM, temperature, draft, ROB, consumption etc., choose the closest marine-standard parameter.

Output: ONLY the parameter `name`.
"""

    response = llm.invoke(prompt)
    return response.content.strip()


def rag_mapping(vectordb, extracted_data):

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        temperature=0,
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )

    final_mapping = {}

    with open("check_keys.txt", "w") as f:
        f.write("EXTRACTED KEY  →  STANDARD PARAMETER\n\n")

    for extracted_key, extracted_value in extracted_data.items():

        # vector DB stores text = "name. description"
        docs = vectordb.similarity_search(extracted_key, k=5)

        # retrieve candidate names from metadata
        candidates = [
            {
                "name": doc.metadata["name"],
                "description": doc.metadata["description"]
            }
            for doc in docs
        ]

        # Only send names to LLM (clean list)
        candidate_names = [c["name"] for c in candidates]

        best_match = rag_map_key(llm, extracted_key, candidate_names)

        with open("check_keys.txt", "a") as f:
            f.write(f"{extracted_key}  →  {best_match}\n")

        # always map: standard_key : value
        # final_mapping[best_match] = extracted_value
        
        if best_match in final_mapping:
            final_mapping[extracted_key] = extracted_value
        else:
            final_mapping[best_match] = extracted_value

    return final_mapping


# -----------------------------
# EXECUTION
# -----------------------------
vectordb = load_param_db()

with open("misuga_parssed4.json", "r") as f:
    parsed_data = json.load(f)

mapped = rag_mapping(vectordb, parsed_data)

with open("mapped_misuga_parssed4.json", "w") as f:
    json.dump(mapped, f, indent=2)

print("Mapping completed.")
