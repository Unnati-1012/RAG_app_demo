# utils/completion.py
import requests
import os

API_KEY = os.getenv("GEMINI_API_KEY")  # Make sure your .env has GEMINI_API_KEY

def generate_completion(prompt, model="gemini-1.5-flash", temperature=0.3):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={API_KEY}"

    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt}
                ]
            }
        ],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": 500
        }
    }

    res = requests.post(url, headers=headers, json=payload)
    data = res.json()

    if "candidates" in data:
        return data["candidates"][0]["content"]["parts"][0]["text"]
    else:
        raise ValueError(f"❌ Unexpected API response: {data}")
