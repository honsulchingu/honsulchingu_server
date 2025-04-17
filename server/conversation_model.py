# %%
# Kernel - base (Python 3.12.7)
# !pip install google-genai
# !pip install keyboard
# !pip install dotenv
import os
import keyboard
import json
from google         import genai
from google.genai   import types
from dotenv         import load_dotenv

# api key 불러오기
load_dotenv()

# prompt.txt 불러오기
with open("prompt.txt", "r", encoding="utf-8") as f:
    prompt_txt = f.read()

# model 설정하기
def init_model():
    client = genai.Client(
        api_key=os.environ.get("GEMINI_API_KEY"),
    )

    model = "gemma-3-27b-it"

    contents = []

    generate_content_config = types.GenerateContentConfig(
        response_mime_type="text/plain",
    )

    contents = response_generate(client,model, contents, generate_content_config, prompt_txt)

    return client, model, contents, generate_content_config

# response 생성하기
def response_generate(client, model, contents, generate_content_config, input_user):
    contents.append(
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=input_user),
            ],
        ),
    )

    response = ""

    for chunk in client.models.generate_content_stream(
        model=model,
        contents=contents,
        config=generate_content_config,
    ):
        if chunk.text is not None:
            print(chunk.text, end="")
            response += chunk.text
    
    contents.append(
        types.Content(
            role="model",
            parts=[
                types.Part.from_text(text=response),
            ],
        ),
    )

    return contents

# model 저장하기
def save_all_to_json(model, contents, generate_content_config, filename="state.json"):
    data = {
        "model": model,

        "contents": [],

        "generate_content_config": {
            "response_mime_type": generate_content_config.response_mime_type,
        },
    }

    data["api_key"] = os.environ.get("GEMINI_API_KEY")

    for content in contents:
        content_data = {
            "role": content.role,
            "parts": [{"text": part.text.strip()} for part in content.parts]
        }
        data["contents"].append(content_data)

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

# model 불러오기
def load_all_from_json(filename="state.json"):
    if not os.path.exists(filename):
        return None, None, None, None

    with open(filename, "r", encoding="utf-8") as f:
        data = json.load(f)

    client = genai.Client(api_key=data["api_key"])

    model = data["model"]

    contents = [
        types.Content(
            role=content["role"],
            parts=[types.Part.from_text(text=part["text"]) for part in content["parts"]]
        )
        for content in data["contents"]
    ]

    generate_content_config = types.GenerateContentConfig(
        response_mime_type=data["generate_content_config"]["response_mime_type"]
    )

    return client, model, contents, generate_content_config

# model 실행하기
if __name__ == "__main__":
    client, model, contents, generate_content_config = load_all_from_json(filename="state.json")

    if not all([client, model, contents, generate_content_config]):
        client, model, contents, generate_content_config = init_model()

    while True:
        input_user = input()

        if keyboard.is_pressed("esc"):
            save_all_to_json(model, contents, generate_content_config, filename="state.json")
            break

        print("\n나: " + input_user, end="\n")
        print("AI: ", end="")
        contents = response_generate(client, model, contents, generate_content_config, input_user)

        print("\n\n=== contents 출력 시작 ===")
        for i, c in enumerate(contents):
            print(f"\n[{i}] role: {c.role}")
            for part in c.parts:
                print(f"{part.text[:100]}")
        print("\n=== contents 출력 종료 ===")

# %%
