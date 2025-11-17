import os
import json
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

DB_DIR = "standard_param_db_orion"

def load_param_db():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return Chroma(
        embedding_function=embeddings,
        persist_directory=DB_DIR
    )


def rag_map_key(llm, extracted_key, candidates):
    prompt = f"""
You are an expert maritime data mapper.

We extracted this key from a vessel report:
Key: "{extracted_key}"

Below are the top candidate standard parameters:

{json.dumps(candidates, indent=2)}

Your task:
- Select ONLY the best matching standard parameter name.
- Return ONLY the selected name, no explanation.
"""

    response = llm.invoke(prompt)
    # print(response)
    return response.content.strip()

with open("check_keys.txt", "w") as f:
            f.write("Keys and best match")

def rag_mapping(vectordb, extracted_data):

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        temperature=0,
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )

    final_mapping = {}

    for key, value in extracted_data.items():
        # Retrieve top-3 matches
        docs = vectordb.similarity_search(key, k=5)

        candidates = [
            doc.metadata["name"]
            for doc in docs
        ]

        # LLM selects best
        best_match = rag_map_key(llm, key, candidates)
        with open("check_keys.txt", "a") as f:
            f.write(f"{key} : {best_match}\n")

        if best_match in final_mapping:
            final_mapping[key] = value
        else:
            final_mapping[best_match] = value

    return final_mapping


vectordb = load_param_db()

# parsed_data 
with open("shanghai_parsed.json", "r") as f:
    parsed_data = json.load(f)

mapped = rag_mapping(vectordb, parsed_data)

with open("shanghai_output3.json", "w") as f:
        f.write(json.dumps(mapped, indent=2))

# print(json.dumps(mapped, indent=2))

