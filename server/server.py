# %%
# Kernel - Python (3.12.7)
# !pip install dotenv
# !pip install fastapi
# !pip install uvicorn
from dotenv             import load_dotenv
from fastapi            import FastAPI
from uvicorn            import run
from pydantic           import BaseModel
from conversation_model import init_setting, init_contents, response_generate, save_setting_to_json, save_contents_to_json, load_setting_from_json, load_contents_from_json
from rds                import init_db, save_contents_to_db, load_contents_from_db, close_db

# 임시 선언하기
client = None
model = None
generate_content_config = None
connection = None
cursor = None
app = FastAPI()

# request 선언하기
class ConversationRequest(BaseModel):
    id_user: str
    select_user: str
    start_user: str
    input_user: str

# startup 구축하기
@app.on_event("startup")
async def startup_event():
    global client, model, generate_content_config, connection, cursor

    load_dotenv()

    client, model, generate_content_config = load_setting_from_json("setting.json")
    connection, cursor = init_db()

    if not all([client, model, generate_content_config]):
        client, model, generate_content_config = init_setting()
    
    save_setting_to_json(model, generate_content_config, "setting.json")

# /conversation_model 구축하기
@app.post("/conversation_model")
async def conversation_model(request: ConversationRequest):
    id_user = request.id_user
    select_user = request.select_user
    start_user = request.start_user
    input_user = request.input_user

    file_name = f"contents_{id_user}_{select_user}_{start_user}.json"
    contents = load_contents_from_json(file_name)

    if not contents:
        contents = load_contents_from_db(cursor, id_user, select_user, start_user, "contents_table")
        
        if not contents:
            contents = init_contents(client, model, generate_content_config, f"prompt_{select_user}.txt", "false")

    contents = response_generate(client, model, generate_content_config, contents, input_user, "true")
    response = contents[-1][0].parts[0].text.strip()

    save_contents_to_json(contents, file_name)
    save_contents_to_db(connection, cursor, id_user, select_user, file_name, "contents_table")
    return {"response": response}

# shutdown 구축하기
@app.on_event("shutdown")
async def shutdown_event():
    close_db(connection, cursor)

# server 실행하기
if __name__ == "__main__":
    run("server:app", host="0.0.0.0", port=8000, reload=True)

# %%
