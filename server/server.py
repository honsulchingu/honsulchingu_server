# %%
# Kernel - Python (3.12.7)
# !pip install fastapi
# !pip install uvicorn
from fastapi                import FastAPI
from uvicorn                import run
from pydantic               import BaseModel
from conversation_model     import init_setting, response_generate
from rds                    import init_db, load_setting_from_db, load_prompt_from_db, save_contents_to_db, load_contents_from_db, load_chat_from_db, load_last_from_db, close_db

# - - - 임시 선언하기 - - - #
client                      = None
model                       = None
generate_content_config     = None
TAG                         = None
connection                  = None
cursor                      = None
app                         = FastAPI()

# - - - request 선언하기 - - - #
class ConversationRequest(BaseModel):
    id_user:            str
    select_user:        str
    input_user:         str
    time_user:          str
    start_user:         str
    shown_user:         str

# - - - startup 구축하기 - - - #
@app.on_event("startup")
async def startup_event():
    global client, model, generate_content_config, TAG, connection, cursor

    connection, cursor = init_db()

    KEY, MODEL, TYPE, TAG = load_setting_from_db(cursor          = cursor,
                                                 table_name      = "setting_table")
    
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
    
    output_ai = CONTENTS[-1][0].parts[0].text.strip()
    time_ai = CONTENTS[-1][1]

    save_contents_to_db(connection      = connection,
                        cursor          = cursor,
                        id_user         = request.id_user,
                        select_user     = request.select_user,
                        start_user      = request.start_user,
                        CONTENTS        = CONTENTS,
                        table_name      = "contents_table")
    
    return {"output_ai": output_ai, "time_ai": time_ai}

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
    LAST = []

    for last in load_last_from_db(
        cursor          = cursor,
        id_user         = request.id_user,
        shown_user      = request.shown_user,
        table_name      = "contents_table"
    ):
        CONTENTS = load_contents_from_db(cursor             = cursor,
                                         id_user            = request.id_user,
                                         select_user        = last["select_user"],
                                         start_user         = last["start"],
                                         table_name         = "contents_table")
        
        CONTENTS = response_generate(client                         = client,
                                     model                          = model,
                                     generate_content_config        = generate_content_config,
                                     input_user                     = TAG,
                                     time_user                      = "",
                                     shown_user                     = "",
                                     CONTENTS                       = CONTENTS)
        
        output_ai = CONTENTS[-1][0].parts[0].text.strip()

        tag = [tag.strip() for tag in output_ai.split(',')][:3]

        LAST.append({
            **last,
            "tag": tag
        })

    return {"last": LAST}

# - - - shutdown 구축하기 - - - #
@app.on_event("shutdown")
async def shutdown_event():
    close_db(connection     = connection,
             cursor         = cursor)

# - - - server 실행하기 - - - #
if __name__ == "__main__":
    run("server:app", host="0.0.0.0", port=8000, reload=True)

# %%
