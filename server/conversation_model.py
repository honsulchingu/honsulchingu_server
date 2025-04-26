# %%
# Kernel - base (Python 3.12.7)
# !pip install google-genai
# !pip install ujson
from os             import environ, path
from google         import genai
from google.genai   import types
from dotenv         import load_dotenv
from ujson          import dump, load
from datetime       import datetime

# setting 설정하기
def init_setting():
    client = genai.Client(api_key=environ.get("GEMINI_API_KEY"))
    model = "gemma-3-27b-it"
    generate_content_config = types.GenerateContentConfig(response_mime_type="text/plain")

    return client, model, generate_content_config

# contents 설정하기
def init_contents(client, model, generate_content_config, prompt_name):
    with open(prompt_name, 'r', encoding="utf-8") as file:
        prompt_txt = file.read()

    contents_with_time = []
    contents_with_time = response_generate(client, model, generate_content_config, contents_with_time, prompt_txt)

    return contents_with_time

# response 생성하기
def response_generate(client, model, generate_content_config, contents_with_time, input_user):
    contents_with_time.append(
        (
            types.Content(
                role = "user",
                
                parts = [types.Part.from_text(text=input_user)]
            ),
            datetime.now().strftime("%Y. %m. %d. %H-%M-%S")
        )
    )

    response = ""

    for chunk in client.models.generate_content_stream(
        model = model,
        contents = [content for content, _ in contents_with_time],
        config = generate_content_config
    ):
        if chunk.text is not None:
            print(chunk.text, end="")
            response += chunk.text
    
    contents_with_time.append(
        (
            types.Content(
                role = "model",
                
                parts = [types.Part.from_text(text=response)]
            ),
            datetime.now().strftime("%Y. %m. %d. %H-%M-%S")
        )
    )

    return contents_with_time

# setting 저장하기 (json)
def save_setting_to_json(model, generate_content_config, file_name):
    data = {
        "model": model,
        
        "generate_content_config": {"response_mime_type": generate_content_config.response_mime_type}
    }
    data["api_key"] = environ.get("GEMINI_API_KEY")
    
    with open(file_name, 'w', encoding="utf-8") as file:
        dump(data, file, ensure_ascii=False, indent=4)

# contents 저장하기 (json)
def save_contents_to_json(contents_with_time, file_name):
    data = {
        "contents": []
    }

    for content, time in contents_with_time:
        content_data = {
            "role": content.role,
            "parts": [{"text": part.text.strip()} for part in content.parts],
            "time": time
        }
        data["contents"].append(content_data)

    with open(file_name, 'w', encoding="utf-8") as file:
        dump(data, file, ensure_ascii=False, indent=4)

# setting 불러오기 (json)
def load_setting_from_json(file_name):
    if not path.exists(file_name):
        return None, None, None

    with open(file_name, 'r', encoding="utf-8") as file:
        data = load(file)

    client = genai.Client(api_key=data["api_key"])
    model = data["model"]
    generate_content_config = types.GenerateContentConfig(response_mime_type=data["generate_content_config"]["response_mime_type"])

    return client, model, generate_content_config

# contents 불러오기 (json)
def load_contents_from_json(file_name):
    if not path.exists(file_name):
        return None

    with open(file_name, 'r', encoding="utf-8") as file:
        data = load(file)

    contents_with_time = [
        (
            types.Content(
                role = content["role"],
                
                parts = [types.Part.from_text(text=part["text"]) for part in content["parts"]]
            ),
            content["time"]
        )
        for content in data["contents"]
    ]

    return contents_with_time

# model 실행하기
if __name__ == "__main__":
    # !pip install keyboard
    from keyboard       import is_pressed

    load_dotenv()
    
    client, model, generate_content_config = load_setting_from_json("setting.json")
    contents = load_contents_from_json("contents.json")
    
    if not all([client, model, generate_content_config, contents]):
        client, model, generate_content_config = init_setting()
        contents = init_contents(client, model, generate_content_config, "prompt.txt")
    
    while True:
        input_user = input()
        
        if is_pressed("esc"):
            save_setting_to_json(model, generate_content_config, "setting.json")
            save_contents_to_json(contents, "contents.json")
            break
        
        print("\n나: " + input_user, end="\n")
        print("AI: ", end="")
        contents = response_generate(client, model, generate_content_config, contents, input_user)
        
        print("\n\n=== contents 출력 시작 ===")
        for i, (content, time) in enumerate(contents):
            print(f"\n[{i}] role: {content.role} ({time})")
            for part in content.parts:
                print(f"{part.text[:100]}")
        print("\n=== contents 출력 종료 ===")

# %%
