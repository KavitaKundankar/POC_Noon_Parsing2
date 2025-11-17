import anthropic
import json
from pathlib import Path
import re
from datetime import datetime

# --------- CONFIG ---------
MODEL_NAME = "anthropic/claude-3-haiku:free"  # or "claude-3-opus-20240229", "claude-3-haiku-20240307"
file_name = "misuga-kaiun"

client = anthropic.Anthropic(api_key=API_KEY)


def extract_key_value_pairs(text: str) -> dict:
    def load_prompt(file_path: str) -> str:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read().strip()

    prompt_text = load_prompt("prompt.txt")
    full_prompt = f"{prompt_text}\n\nEMAIL CONTENT:\n{text}"

    # ---- Claude API Request ----
    response = client.messages.create(
        model=MODEL_NAME,
        max_tokens=4096,
        messages=[
            {"role": "user", "content": full_prompt}
        ],
        system="You extract structured JSON from text."
    )

    # Extract response text
    raw_text = response.content[0].text
    json_clean = re.sub(r"^```json\s*|\s*```$", "", raw_text.strip())
    data = json.loads(json_clean)

    # Print token usage
    print("\n---- TOKEN USAGE ----")
    print(f"Input Tokens:  {response.usage.input_tokens}")
    print(f"Output Tokens: {response.usage.output_tokens}")

    # Save JSON Output
    filename = str(datetime.now()) + "_" + file_name + "_extracted_claude"
    print("\nSaving output as:", filename + ".json")

    with open(filename + ".json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    print("\n✅ Extraction Completed")
    return {"raw_output": raw_text}


# ---- Read Input File ----
input_path = Path(f"input/{file_name}.txt")
email_text = input_path.read_text(encoding="utf-8")

extract_key_value_pairs(email_text)
