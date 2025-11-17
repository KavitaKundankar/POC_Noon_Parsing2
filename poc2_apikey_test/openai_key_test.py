from openai import OpenAI
import json
from pathlib import Path
import re
from datetime import datetime

# --------- CONFIG ---------   
# # Change based on your provider
file_name = "misuga-kaiun"

# If using OPENROUTER, uncomment this:
# client = OpenAI(api_key=API_KEY, base_url="https://openrouter.ai/api/v1")

# If using OPENAI directly, use this:
client = OpenAI(api_key=API_KEY)


def extract_key_value_pairs(text: str) -> dict:
    def load_prompt(file_path: str) -> str:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read().strip()

    prompt_text = load_prompt("prompt.txt")
    full_prompt = f"{prompt_text}\n\nEMAIL CONTENT:\n{text}"

    # ---- OpenAI API Call ----
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "You extract structured JSON from text."},
            {"role": "user", "content": full_prompt}
        ],
    )

    # Extract response text
    raw_text = response.choices[0].message.content
    json_clean = re.sub(r"^```json\s*|\s*```$", "", raw_text.strip())
    data = json.loads(json_clean)

    # Print token usage if available
    print("\n---- TOKEN USAGE ----")
    try:
        print(response.usage)
    except:
        print("Usage details not available for this provider")

    # Save output JSON
    filename = str(datetime.now()) + "_" + file_name + "extracted_openai"
    print(filename)
    with open(filename +  ".json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


    print("\nExtraction Completed")
    return {"raw_output": raw_text}


# ---- Read Input File ----
input_path = Path(f"input/{file_name}.txt")
email_text = input_path.read_text(encoding="utf-8")

extract_key_value_pairs(email_text)
