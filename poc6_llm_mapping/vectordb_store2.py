import json
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

PARAM_FILE = "misuga_standard_parameters_with_description.json"
DB_DIR = "standard_param_db_misuga_description"

def build_standard_param_db():
    with open(PARAM_FILE, "r") as f:
        params = json.load(f)

    texts = []
    metadatas = []

    for item in params:
        name = item.get("name", "")
        desc = item.get("description", "")

        # Text used for embeddings
        combined_text = f"{name}. {desc}"

        texts.append(combined_text)
        metadatas.append({
            "name": name,
            "description": desc
        })

    # Embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Create vector DB
    vectordb = Chroma.from_texts(
        texts=texts,
        metadatas=metadatas,
        embedding=embeddings,
        persist_directory=DB_DIR
    )

    vectordb.persist()
    print(f"Vector DB created with {len(texts)} standard parameters.")
    return vectordb


if __name__ == "__main__":
    build_standard_param_db()
