import os
from fastapi import FastAPI, Query
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI

# Load .env
load_dotenv(dotenv_path="../../Personal_AI_Trip_Planner/keys.env")

Kimi_K2_HF_Token = os.getenv("Kimi_K2_HF_Token")
Kimi_K2_HF_Base  = os.getenv("Kimi_K2_HF_Base")
Kimi_K2_HF_Model = os.getenv("Kimi_K2_HF_Model", "moonshotai/Kimi-K2-Instruct:fireworks-ai")

app = FastAPI()

class PromptRequest(BaseModel):
    prompt: str

class KimiK2Client:
    def __init__(self, base_url: str, token: str, model: str):
        self.client = OpenAI(base_url=base_url, api_key=token)
        self.model = model

    def get_response(self, prompt: str) -> str:
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
        )
        return completion.choices[0].message.content

kimi_client = KimiK2Client(Kimi_K2_HF_Base, Kimi_K2_HF_Token, Kimi_K2_HF_Model)

# ---- helper you can call directly in Python
def generate(prompt: str) -> str:
    result = kimi_client.get_response(prompt)
    print(f"> Prompt: {prompt}\n< Response: {result}")
    return result

# ---- POST endpoint (JSON body)
@app.post("/chat")
def chat(request: PromptRequest):
    return {"response": generate(request.prompt)}

# ---- GET endpoint (?prompt=...)
@app.get("/chat")
def chat_get(prompt: str = Query(..., description="User prompt")):
    return {"response": generate(prompt)}

