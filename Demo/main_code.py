import google.generativeai as genai
import json
import pandas as pd
from pathlib import Path
from pydantic import BaseModel, Field
import re
from datetime import datetime

from dotenv import load_dotenv
import os

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")


genai.configure(api_key=API_KEY)
file_name = "misuga-kaiun"

class VerificationResult(BaseModel):
    is_successful: bool = Field(..., description="True if extraction is valid and correctly formatted")
    reasoning: str = Field(..., description="Explain why the result passed or failed verification")

def load_prompt(file_path: str) -> str:
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read().strip()
    
def execute_extraction(model, prompt_text, email_text):
    full_prompt = f"{prompt_text}\n\nEMAIL CONTENT:\n{email_text}"
    response = model.generate_content(full_prompt)
    return response.text.strip()

def verify_output(model, extracted_text):
    verifier_prompt = f"""
You are a verifier that checks if the given model output is valid.
Check if:
1. The format is correct key-value pairs or JSON.
2. All keys have non-empty values.
3. It makes sense as parsed email data.
4. all values from input are present in output , no value from input is missing
5. vessal name and voyage no is present in output if it is given in input

Return your decision in JSON with:
- is_successful (true/false)
- reasoning (short explanation)

OUTPUT TO VERIFY:
{extracted_text}
"""

    response = model.generate_content(verifier_prompt)

    try:
        json_str = response.text.strip().replace("```json", "").replace("```", "")
        result = json.loads(json_str)
        return VerificationResult(**result)
    except Exception:
        return VerificationResult(is_successful=False, reasoning="Invalid verification output format")

def pev_pipeline(email_text, max_retries=2):
    executor_model = genai.GenerativeModel("models/gemini-2.5-flash-lite")
    verifier_model = genai.GenerativeModel("models/gemini-2.5-flash-lite")

    prompt_text = load_prompt("prompt.txt")

    for attempt in range(max_retries):
        print(f"\n Attempt {attempt+1} - Executing extraction...")
        extracted_text = execute_extraction(executor_model, prompt_text, email_text)

        print("Verifying output...")
        verification = verify_output(verifier_model, extracted_text)

        print(f"Verifier result: {verification.is_successful} — {verification.reasoning}")
        if verification.is_successful:
            print("Verified successfully!")
            return extracted_text

        print("Verification failed. Retrying with correction prompt...")
        prompt_text += f"\n\nThe previous output failed verification because: {verification.reasoning}. Please fix and regenerate."

    print("Max retries reached. Returning last output.")
    return extracted_text

if __name__ == "__main__":
    input_path = Path(f"{file_name}.txt")
    email_text = input_path.read_text(encoding="utf-8")

    final_output = pev_pipeline(email_text)

    response_in_json = re.sub(r"^```json\s*|\s*```$", "", final_output.strip())
    data = json.loads(response_in_json)

    #write data to json file
    filename = f"extracted_{file_name}_{str(datetime.now())}" 
    with open(filename +  ".json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
