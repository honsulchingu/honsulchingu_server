# %%
# Kernel - Python (3.12.7)
# !pip install fastapi
# !pip install uvicorn
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from conversation_model import init_model, response_generate, save_all_to_json, load_all_from_json

# 임시 선언하기
client = None

model = None

contents = None

generate_content_config = None

app = FastAPI()

# request 선언하기
class ConversationRequest(BaseModel):
    input_user: str

# startup 구축하기
@app.on_event("startup")
async def startup_event():
    global client, model, contents, generate_content_config

    client, model, contents, generate_content_config = load_all_from_json(filename="state.json")

    if not all([client, model, contents, generate_content_config]):
        client, model, contents, generate_content_config = init_model()

# /conversation_model 구축하기
@app.post("/conversation_model")
async def conversation_model(request: ConversationRequest):
    global contents

    input_user = request.input_user

    contents = response_generate(client, model, contents, generate_content_config, input_user)

    response = contents[-1].parts[0].text.strip()

    save_all_to_json(model, contents, generate_content_config, filename="state.json")

    return {"response": response}

# server 실행하기
if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)

# %%
