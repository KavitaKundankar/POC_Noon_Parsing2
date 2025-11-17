import google.generativeai as genai
import json
from pathlib import Path
import re
from datetime import datetime
from openai import OpenAI 


genai.configure(api_key=API_KEY)
file_name= "misuga-kaiun"

def extract_key_value_pairs(text: str) -> dict:

    def load_prompt(file_path: str) -> str:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read().strip()

    prompt_text = load_prompt("prompt.txt")

    full_prompt = f"{prompt_text}\n\nEMAIL CONTENT:\n{text}"

    # response = model.generate_content(full_prompt).text

    # using openrouter for testing api keys

    
    client = OpenAI(
        api_key = API_KEY,
        base_url = "https://openrouter.ai/api/v1",
    )


    response = client.chat.completions.create(
    model="deepseek/deepseek-r1:free",
    messages=[
        {"role": "user", "content": full_prompt}
    ]
    )

    print(response)

    response_in_json = re.sub(r"^```json\s*|\s*```$", "", response.strip())
    data = json.loads(response_in_json)
    

    #write data to json file
    filename = str(datetime.now()) + "_" + file_name + "extracted"
    print(filename)
    with open(filename +  ".json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    raw_text = response.strip()
    print("Returning raw response.")
    return {"raw_output": raw_text}


input_path = Path(f"input/{file_name}.txt")
email_text = input_path.read_text(encoding="utf-8")
extract_key_value_pairs(email_text)