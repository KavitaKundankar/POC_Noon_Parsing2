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
    memori = load_memori()

    with open("check_keys.txt", "w") as f:
        f.write("EXTRACTED KEY  →  STANDARD PARAMETER\n\n")

    for extracted_key, extracted_value in extracted_data.items():

        # -----------------------------------------
        # 1. CHECK MEMORY FIRST
        # -----------------------------------------
        if extracted_key in memori:
            best_match = memori[extracted_key]
            confidence = 1.0  # From memory → treat as fully trusted
            print(f"[MEMORI] {extracted_key} → {best_match}")
        
        else:
            # -----------------------------------------
            # 2. VECTOR DB → LLM mapping
            # -----------------------------------------
            # docs = vectordb.similarity_search(extracted_key, k=5)
            # print(docs)

            # candidates = [
            #     doc.metadata["name"]
            #     for doc in docs
            # ]

            docs = vectordb.similarity_search_with_score(extracted_key, k=5)

            candidates = []
            candidate_names = []

            for doc, score in docs:
                name = doc.metadata.get("name")
                desc = doc.metadata.get("description")

                # convert score → confidence (distance → similarity)
                confidence = max(0.0, 1 - score)

                candidates.append({
                    "name": name,
                    "description": desc,
                    "distance": score,
                    "vconfidence": confidence
                })

                candidate_names.append(name)


            best_match = rag_map_key(llm, extracted_key, candidates)
            confidence = next((c["vconfidence"] for c in candidates if c["name"] == best_match),0.0)

            print(f"[LLM] {extracted_key} → {best_match} ({confidence})")

            # -----------------------------------------
            # 3. IF CONFIDENCE HIGH → SAVE TO MEMORY
            # -----------------------------------------
            if confidence >= 0.45:
                memori[extracted_key] = best_match
                save_memori(memori)
                print(f"[MEMORI-SAVED] {extracted_key} → {best_match}")

        # -----------------------------------------
        # Final mapping dictionary
        # -----------------------------------------
        final_mapping[best_match] = extracted_value

        # Write to check file
        with open("check_keys.txt", "a") as f:
            f.write(f"{extracted_key}  →  {best_match}\n")

    return final_mapping


MEMORI_FILE = "misuga_memori2.json"

def load_memori():
    """Load memory mapping file."""
    if not os.path.exists(MEMORI_FILE):
        return {}
    with open(MEMORI_FILE, "r") as f:
        return json.load(f)

def save_memori(mem):
    """Save updated memory mapping."""
    with open(MEMORI_FILE, "w") as f:
        json.dump(mem, f, indent=2)

# -----------------------------
# EXECUTION
# -----------------------------
vectordb = load_param_db()

with open("misuga_parsed4.json", "r") as f:
    parsed_data = json.load(f)

mapped = rag_mapping(vectordb, parsed_data)

with open("mapped_misuga_parssed4.json", "w") as f:
    json.dump(mapped, f, indent=2)

print("Mapping completed.")
