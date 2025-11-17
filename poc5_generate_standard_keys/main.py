import google.generativeai as genai
import json
import os
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

MODEL = "gemini-1.5-flash"   # or your preferred model

def generate_description(key, prompt):
    """Generate description for a single key."""
    full_prompt = f"{prompt}\n\nParameter: {key}"
    response = genai.GenerativeModel(MODEL).generate_content(full_prompt)
    return response.text.strip()

def process_keys(keys, prompt):
    result = []
    for key in keys:
        description = generate_description(key, prompt)
        result.append({
            "standard_parameter": key,
            "description": description,
            "variant": []   # You can change this later
        })
    return result

# Example usage
if __name__ == "__main__":
    keys = ["Voyage_Number", "Start_Port", "Fuel_Consumption"]  # replace later
    prompt = "Write a clear technical description for this vessel standard parameter."

    output = process_keys(keys, prompt)

    with open("standard_parameters.json", "w") as f:
        json.dump(output, f, indent=4)

    print("Generated JSON saved!")
