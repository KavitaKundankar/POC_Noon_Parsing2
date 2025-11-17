import json
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from pathlib import Path

PARAM_FILE = "orion_standard_parameters.json"
DB_DIR = "standard_param_db_orion"

def build_standard_param_db():
    # Load list of standard parameters
    with open(PARAM_FILE, "r") as f:
        params = json.load(f)

    # Create Document list with descriptions = same as name (no description available)
    docs = [
        (param, {"name": param})
        for param in params
    ]

    # Embedding model
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Convert to Chroma format
    vectordb = Chroma.from_texts(
        texts=[d[0] for d in docs],
        metadatas=[d[1] for d in docs],
        embedding=embeddings,
        persist_directory=DB_DIR
    )

    vectordb.persist()
    print(f"Vector DB created with {len(params)} standard parameters.")
    return vectordb

if __name__ == "__main__":
    build_standard_param_db()
