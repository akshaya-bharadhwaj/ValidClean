import requests

LM_STUDIO_URL = "http://127.0.0.1:1234/v1/chat/completions"

import re

def extract_json(text):
    # Remove markdown ```json ``` fences
    text = re.sub(r"```json|```", "", text)
    
    # Extract first JSON object found
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return match.group(0)
    return text.strip()


def call_llm(prompt, temperature = 0.2):
    payload = {
        "model":"phi-3-mini-4k-instruct",
        "messages":[
            {"role":"system", "content": "You are an expert data scientist designing data cleaning pipelines."},
            {"role":"user", "content":prompt}
        ],
        "temperature": temperature,
        "max_tokens": 500
    }
    
    response = requests.post(LM_STUDIO_URL, json=payload, timeout=600)
    
    if response.status_code != 200:
        print("LM Studio Error:", response.text)
    
    response.raise_for_status()
    data = response.json()
    
    
    
    
    
    # Handle Possible LM Studio response formats
    if "choices" in data:
        raw = data["choices"][0]["message"]["content"]
        return extract_json(raw)
    else:
        print("Unexpected LM Studio response:", data)
        raise ValueError("Unknown response format from LM Studio")
