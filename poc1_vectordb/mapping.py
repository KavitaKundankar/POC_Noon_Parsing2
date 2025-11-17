import json
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

DATA_DIR = "chroma_store2"

def load_vector_db():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectordb = Chroma(
        persist_directory=DATA_DIR,
        embedding_function=embeddings
    )

    print("Loaded existing Chroma Vector DB.")
    return vectordb

def map_to_standard_keys(vectordb, extracted_data):
    """Map extracted keys to the closest standard parameter using vector similarity"""
    retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 1})
    mapped_result = {}

    for key, value in extracted_data.items():
        docs = retriever.get_relevant_documents(key)
        print(f"docs are given",docs)
        if docs:
            best_match = docs[0].metadata["name"]
            mapped_result[best_match] = value
        else:
            mapped_result[key] = value 

    with open("mapping_file", "w") as f:
        f.write("Mapping is given below : " )
        f.write(json.dumps(mapped_result, indent=2))
    return mapped_result

vectordb = load_vector_db()
json_str = response.strip().replace("```json", "").replace("```", "")
data = json.loads(json_str)
mapped = map_to_standard_keys(vectordb, extracted)