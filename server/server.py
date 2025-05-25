# %%
# Kernel - Python (3.12.7)
# !pip install fastapi
# !pip install uvicorn
from fastapi                import FastAPI
from uvicorn                import run
from pydantic               import BaseModel
from conversation_model     import init_setting, response_generate
from rds                    import (init_db,        load_setting_from_db,       save_contents_to_db,        add_user_to_db,             load_character_from_db,
                                    close_db,       load_prompt_from_db,        load_contents_from_db,      delete_user_from_db,        load_chat_from_db,
                                                                                                                                        load_last_from_db)

# - - - 임시 선언하기 - - - #
client                      = None
model                       = None
generate_content_config     = None
KAKAO                       = None
BEGIN                       = None
TAG                         = None
connection                  = None
cursor                      = None
app                         = FastAPI()

# - - - ConversationRequest 선언하기 - - - #
class ConversationRequest(BaseModel):
    id_user:            str
    select_user:        str
    input_user:         str
    time_user:          str
    start_user:         str
    shown_user:         str

# - - - ManagementRequest 선언하기 - - - #
class ManagementRequest(BaseModel):
    email:          str
    nickname:       str
    image:          str
    startday:       str

# - - - startup 구축하기 - - - #
@app.on_event("startup")
async def startup_event():
    global client, model, generate_content_config, KAKAO, BEGIN, TAG, connection, cursor
    
    connection, cursor = init_db()
    
    KEY, MODEL, TYPE, KAKAO, BEGIN, TAG = load_setting_from_db(cursor           = cursor,
                                                               table_name       = "setting_table")
    
    client, model, generate_content_config = init_setting(KEY       = KEY,
                                                          MODEL     = MODEL,
                                                          TYPE      = TYPE)

# - - - /conversation_model 구축하기 - - - #
@app.post("/conversation_model")
async def conversation_model(request: ConversationRequest):
    CONTENTS = load_contents_from_db(cursor             = cursor,
                                     id_user            = request.id_user,
                                     select_user        = request.select_user,
                                     start_user         = request.start_user,
                                     table_name         = "contents_table")
    
    if not CONTENTS:
        PROMPT = load_prompt_from_db(cursor             = cursor,
                                     select_user        = request.select_user,
                                     table_name         = "prompt_table")
        
        CONTENTS = response_generate(client                         = client,
                                     model                          = model,
                                     generate_content_config        = generate_content_config,
                                     input_user                     = PROMPT,
                                     time_user                      = request.time_user,
                                     shown_user                     = "false",
                                     CONTENTS                       = [])
        
    CONTENTS = response_generate(client                         = client,
                                 model                          = model,
                                 generate_content_config        = generate_content_config,
                                 input_user                     = request.input_user,
                                 time_user                      = request.time_user,
                                 shown_user                     = request.shown_user,
                                 CONTENTS                       = CONTENTS)
    
    output_ai = CONTENTS[-1][0].parts[0].text
    time_ai = CONTENTS[-1][1]
    
    save_contents_to_db(connection      = connection,
                        cursor          = cursor,
                        id_user         = request.id_user,
                        select_user     = request.select_user,
                        start_user      = request.start_user,
                        CONTENTS        = CONTENTS,
                        table_name      = "contents_table")
    
    return {"output_ai": output_ai, "time_ai": time_ai}

# - - - /load_setting 구축하기 - - - #
@app.post("/load_setting")
async def load_setting():
    return {"kakao": KAKAO, "begin": BEGIN, "tag": TAG}

# - - - /load_chat 구축하기 - - - #
@app.post("/load_chat")
async def load_chat(request: ConversationRequest):
    CHAT = load_chat_from_db(cursor             = cursor,
                             id_user            = request.id_user,
                             select_user        = request.select_user,
                             start_user         = request.start_user,
                             shown_user         = request.shown_user,
                             table_name         = "contents_table")
    
    return {"chat": CHAT}

# - - - /load_last 구축하기 - - - #
@app.post("/load_last")
async def load_last(request: ConversationRequest):
    LAST = load_last_from_db(cursor         = cursor,
                             id_user        = request.id_user,
                             shown_user     = request.shown_user,
                             table_name     = "contents_table")
    
    return {"last": LAST}

# - - - /create_tag 구축하기 - - - #
@app.post("/create_tag")
async def create_tag(request: ConversationRequest):
    CONTENTS = load_contents_from_db(cursor             = cursor,
                                     id_user            = request.id_user,
                                     select_user        = request.select_user,
                                     start_user         = request.start_user,
                                     table_name         = "contents_table")
    
    CONTENTS = response_generate(client                         = client,
                                 model                          = model,
                                 generate_content_config        = generate_content_config,
                                 input_user                     = request.input_user,
                                 time_user                      = "",
                                 shown_user                     = "",
                                 CONTENTS                       = CONTENTS)
    
    tag = ['#' + tag.strip() for tag in CONTENTS[-1][0].parts[0].text.split(',')][:3]
    
    return {"tag": tag}

# - - - /load_character 구축하기 - - - #
@app.post("/load_character")
async def load_character():
    CHARACTER = load_character_from_db(cursor           = cursor,
                                       table_name       = "prompt_table")
    
    return {"character": CHARACTER}

# - - - /add_user 구축하기 - - - #
@app.post("/add_user")
async def add_user(request: ManagementRequest):
    add_user_to_db(connection       = connection,
                   cursor           = cursor,
                   email            = request.email,
                   nickname         = request.nickname,
                   image            = request.image,
                   startday         = request.startday,
                   table_name       = "user_table")

# - - - /delete_user 구축하기 - - - #
@app.post("/delete_user")
async def delete_user(request: ManagementRequest):
    delete_user_from_db(connection          = connection,
                        cursor              = cursor,
                        email               = request.email,
                        table_name_1        = "user_table",
                        table_name_2        = "contents_table")

# - - - shutdown 구축하기 - - - #
@app.on_event("shutdown")
async def shutdown_event():
    close_db(connection     = connection,
             cursor         = cursor)

# - - - server 실행하기 - - - #
if __name__ == "__main__":
    run("server:app", host="0.0.0.0", port=8000, reload=True)

# %%
