import json
from pathlib import Path
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
import os
# from pydantic import BaseModel


DATA_DIR = Path("chroma_store_misuga")
PARAM_FILE = Path(".json")
INPUT_FILE = Path("input.txt")
OUTPUT_FILE = Path("_output.txt")


def build_vector_db():
    with open(PARAM_FILE, "r") as f:
        params = json.load(f)

    # docs = [Document(page_content=p["description"], metadata={"name": p["name"]}) for p in params]
    docs = [
    Document(
        page_content=f"{p['name']} - {p['description']}",
        metadata={"name": p["name"]}
    )
    for p in params
    ]

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    vectordb = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=str(DATA_DIR)
    )
    vectordb.persist()
    print(f"Chroma Vector DB initialized with {len(docs)} parameters.")
    return vectordb


def extract_key_values(text: str):
    """Use Gemini LLM to extract structured data (key-val pairs)"""
    prompt_template = """
    You are a data extraction assistant.
    Extract measurable information from the given text and return valid JSON key-value pairs.
    Rules:
    - Each line may contain one pairs.
    - Keys and values are separated by ":" 
    - All keys must be **flattened** — do not use nested structures.
    - each result have same format
    - keys format is same for all
    - keys should be meaningful
    - output will be single json
    - all values from input are present in output , no value from input is missing
    
    Example:
    Input: "The ship draft is 5.2m and power is 1500kW."
    Output: "draft": 5.2, "power": 1500

    Text: {input_text}
    Output:
    """


    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0, google_api_key=os.getenv("GOOGLE_API_KEY"))
    prompt = PromptTemplate(template=prompt_template, input_variables=["input_text"])
    chain = LLMChain(llm=llm, prompt=prompt)

    response = chain.run(input_text=text)
    # print("\n Extracted Key-Val Pairs:")
    print(f"llm response will be : ", response)

    try:
        json_str = response.strip().replace("```json", "").replace("```", "")
        data = json.loads(json_str)
        with open(OUTPUT_FILE, "w") as f:
            f.write("Parsing ia given below : ")
            f.write(json.dumps(data, indent=2))
    except json.JSONDecodeError:
        data = {}
        # print("Exception")
    return data

def map_to_standard_keys(vectordb, extracted_data):
    """Map extracted keys to the closest standard parameter using vector similarity"""
    retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 1})
    mapped_result = {}

    for key, value in extracted_data.items():
        docs = retriever.get_relevant_documents(key)
        print(f"key will be",key)

        print(f"docs are given",docs)
        if docs:
            best_match = docs[0].metadata["name"]
            mapped_result[best_match] = value
        else:
            mapped_result[key] = value 

    with open(OUTPUT_FILE, "a") as f:
        f.write("Mapping is given below : " )
        f.write(json.dumps(mapped_result, indent=2))
    return mapped_result



def main():

    vectordb = build_vector_db()

    input_text = INPUT_FILE.read_text()

    extracted = extract_key_values(input_text)
    # with open("shanghai.json", "r") as f:
    #     extracted = json.load(f)

    mapped = map_to_standard_keys(vectordb, extracted)

    print("\n Final Mapped Output:")
    print(json.dumps(mapped, indent=2))


if __name__ == "__main__":
    main()



















# def map_to_standard_keys(vectordb, extracted_data):
#     """Map extracted keys to the closest standard parameter using vector similarity + include similarity score"""
    
#     mapped_result = {}

#     for key, value in extracted_data.items():
#         # get docs WITH score
#         results = vectordb.similarity_search_with_score(key, k=1)
#         if not results:
#             mapped_result[key] = {
#                 "value": value,
#                 "similarity_score": None
#             }
#             continue

#         best_doc, distance = results[0]
#         similarity = 1 - distance   # (0–1 scale roughly)

#         standard_key = best_doc.metadata["name"]

#         mapped_result[standard_key] = {
#             "value": value,
#             "similarity_score": round(similarity, 4)
#         }

#     # write output
#     with open(OUTPUT_FILE, "a") as f:
#         f.write("\n\nMapping with similarity scores:\n")
#         f.write(json.dumps(mapped_result, indent=2))

#     return mapped_result