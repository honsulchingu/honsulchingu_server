# %%
# .py3127_env\Scripts\activate && pip install uvicorn fastapi python-multipart faster-whisper tensorflow-cpu
from uvicorn                            import run
from pydantic                           import BaseModel
from fastapi                            import FastAPI, UploadFile, File, Form; app = FastAPI()
from faster_whisper                     import WhisperModel
from conversation.ConversationModel     import init_conversation, response_generate
from analyzation.AnalyzationModel       import init_cnn, judgement_generate, tts_generate
from aws.RdsManager                     import RdsManager

# 공유변수 임시 선언하기
client                      = None
model                       = None
generate_content_config     = None
whisper                     = None
cnn_men                     = None
cnn_women                   = None
TTS                         = None
KAKAO                       = None
GOOGLE                      = None
BEGIN                       = None
TAG                         = None
connection                  = None
cursor                      = None
rds                         = None

# ConversationRequest 선언하기
class ConversationRequest(BaseModel):
    id_user:            str
    select_user:        str
    input_user:         str
    time_user:          str
    start_user:         str

# ManagementRequest 선언하기
class ManagementRequest(BaseModel):
    email:          str
    nickname:       str
    image:          str
    age:            str
    gender:         str
    startday:       str

# startup 구축하기
@app.on_event("startup")
async def startup_event():
    global client, model, generate_content_config, whisper, cnn_men, cnn_women, TTS, KAKAO, GOOGLE, BEGIN, TAG, connection, cursor, rds
    
    rds = RdsManager()
    
    connection, cursor = rds.init_db()
    
    KEY, MODEL, TYPE, TTS, KAKAO, GOOGLE, BEGIN, TAG = rds.load_setting_from_db(cursor          = cursor,
                                                                                table_name      = "setting_table")
    
    client, model, generate_content_config = init_conversation(KEY          = KEY,
                                                               MODEL        = MODEL,
                                                               TYPE         = TYPE)
    
    whisper = WhisperModel("small", device = "cpu", compute_type = "int16")
    
    cnn_men = init_cnn()
    cnn_men.load_weights("/home/ubuntu/honsulchingu_server/analyzation/cnn_men.weights.h5")
    
    cnn_women = init_cnn()
    cnn_women.load_weights("/home/ubuntu/honsulchingu_server/analyzation/cnn_women.weights.h5")

# /conversation_model 구축하기
@app.post("/conversation_model")
async def conversation_model(request: ConversationRequest):
    CONTENTS = rds.load_contents_from_db(cursor             = cursor,
                                         id_user            = request.id_user,
                                         select_user        = request.select_user,
                                         start_user         = request.start_user,
                                         table_name         = "contents_table")
    
    if not CONTENTS:
        PROMPT = rds.load_prompt_from_db(cursor             = cursor,
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
                                 shown_user                     = "true",
                                 CONTENTS                       = CONTENTS)
    
    output_ai = CONTENTS[-1][0].parts[0].text

    time_ai = CONTENTS[-1][1]
    
    rds.save_contents_to_db(connection      = connection,
                            cursor          = cursor,
                            id_user         = request.id_user,
                            select_user     = request.select_user,
                            start_user      = request.start_user,
                            CONTENTS        = CONTENTS,
                            table_name      = "contents_table")
    
    return {"output_ai": output_ai, "time_ai": time_ai}

# /analyzation_model 구축하기
@app.post("/analyzation_model")
async def analyzation_model(id_user:           str = Form(...),
                            select_user:       str = Form(...),
                            speak_user:        str = Form(...),
                            time_user:         str = Form(...),
                            start_user:        str = Form(...),
                            wav_user:          UploadFile = File(...)):
    
    JUDGEMENT, SENTENCE = judgement_generate(whisper        = whisper,
                                             cnn            = cnn_men, # 수정사항 3 (3/5) cnn_men 변경
                                             wav_bytes      = await wav_user.read())
    
    print(f"JUDGEMENT: {JUDGEMENT}") # 수정사항 4 (4/5) print() 추가
    
    CONTENTS =  rds.load_contents_from_db(cursor            = cursor,
                                          id_user           = id_user,
                                          select_user       = select_user,
                                          start_user        = start_user,
                                          table_name        = "contents_table")
    
    judgement = rds.load_judgement_from_db(cursor           = cursor,
                                           id_user          = id_user,
                                           start_user       = start_user,
                                           table_name       = "contents_table")
    
    print(f"judgement: {judgement}") # 수정사항 5 (4/5) print() 추가
    
    if not CONTENTS:
        PROMPT = rds.load_prompt_from_db(cursor             = cursor,
                                         select_user        = select_user,
                                         table_name         = "prompt_table")
        
        CONTENTS = response_generate(client                         = client,
                                     model                          = model,
                                     generate_content_config        = generate_content_config,
                                     input_user                     = PROMPT,
                                     time_user                      = time_user,
                                     shown_user                     = "false",
                                     CONTENTS                       = [])
        
    if judgement == "0%":
        CONTENTS = response_generate(client                         = client,
                                     model                          = model,
                                     generate_content_config        = generate_content_config,
                                     input_user                     = "사용자가 0% 취했습니다. 당신도 0% 취했습니다. 이에 맞게 말투를 변경합니다. 메모리에 탑재가 완료되었다면 대답은 '　'로 대답합니다.",
                                     time_user                      = time_user,
                                     shown_user                     = "false",
                                     CONTENTS                       = CONTENTS)
        
    if judgement == "25%":
        CONTENTS = response_generate(client                         = client,
                                     model                          = model,
                                     generate_content_config        = generate_content_config,
                                     input_user                     = "사용자가 25% 취했습니다. 당신도 25% 취했습니다. 이에 맞게 말투를 변경합니다. 메모리에 탑재가 완료되었다면 대답은 '　'로 대답합니다.",
                                     time_user                      = time_user,
                                     shown_user                     = "false",
                                     CONTENTS                       = CONTENTS)
        
    if judgement == "50%":
        CONTENTS = response_generate(client                         = client,
                                     model                          = model,
                                     generate_content_config        = generate_content_config,
                                     input_user                     = "사용자가 50% 취했습니다. 당신도 50% 취했습니다. 이에 맞게 말투를 변경합니다. 메모리에 탑재가 완료되었다면 대답은 '　'로 대답합니다.",
                                     time_user                      = time_user,
                                     shown_user                     = "false",
                                     CONTENTS                       = CONTENTS)
        
    if judgement == "75%":
        CONTENTS = response_generate(client                         = client,
                                     model                          = model,
                                     generate_content_config        = generate_content_config,
                                     input_user                     = "사용자가 75% 취했습니다. 당신도 75% 취했습니다. 이에 맞게 말투를 변경합니다. 메모리에 탑재가 완료되었다면 대답은 '　'로 대답합니다.",
                                     time_user                      = time_user,
                                     shown_user                     = "false",
                                     CONTENTS                       = CONTENTS)
        
    if judgement == "100%":
        CONTENTS = response_generate(client                         = client,
                                     model                          = model,
                                     generate_content_config        = generate_content_config,
                                     input_user                     = "사용자가 100% 취했습니다. 당신도 100% 취했습니다. 이에 맞게 말투를 변경합니다. 메모리에 탑재가 완료되었다면 대답은 '　'로 대답합니다.",
                                     time_user                      = time_user,
                                     shown_user                     = "false",
                                     CONTENTS                       = CONTENTS)
        
    CONTENTS = response_generate(client                         = client,
                                 model                          = model,
                                 generate_content_config        = generate_content_config,
                                 input_user                     = SENTENCE,
                                 time_user                      = time_user,
                                 shown_user                     = "true",
                                 CONTENTS                       = CONTENTS)
    
    print(CONTENTS[-2][0].parts[0].text) # 수정사항 2 (2/5) print() 추가
    
    output_ai = CONTENTS[-1][0].parts[0].text
    
    tts_ai = tts_generate(client            = client,
                          tts               = TTS,
                          speak_user        = speak_user,
                          speak_ai          = "(기본적으로 약간 빠르게 말하는 사람처럼)",
                          output_ai         = output_ai)
    
    rds.save_contents_to_db(connection      = connection,
                            cursor          = cursor,
                            id_user         = id_user,
                            select_user     = select_user,
                            start_user      = start_user,
                            CONTENTS        = CONTENTS,
                            table_name      = "contents_table")
    
    rds.add_judgement_to_db(connection      = connection,
                            cursor          = cursor,
                            role            = "user",
                            id_user         = id_user,
                            time_user       = time_user,
                            judgement       = JUDGEMENT,
                            table_name      = "contents_table")
    
    return {"output_ai": output_ai, "tts_ai": tts_ai}

# /load_setting 구축하기
@app.post("/load_setting")
async def load_setting():
    return {"kakao": KAKAO, "google": GOOGLE, "begin": BEGIN, "tag": TAG}

# /load_character 구축하기
@app.post("/load_character")
async def load_character():
    CHARACTER = rds.load_character_from_db(cursor           = cursor,
                                           table_name       = "prompt_table")
    
    return {"character": CHARACTER}

# /load_last 구축하기
@app.post("/load_last")
async def load_last(request: ConversationRequest):
    LAST = rds.load_last_from_db(cursor         = cursor,
                                 id_user        = request.id_user,
                                 table_name     = "contents_table")
    
    return {"last": LAST}

# /load_chat 구축하기
@app.post("/load_chat")
async def load_chat(request: ConversationRequest):
    CHAT = rds.load_chat_from_db(cursor             = cursor,
                                 id_user            = request.id_user,
                                 select_user        = request.select_user,
                                 start_user         = request.start_user,
                                 table_name         = "contents_table")
    
    return {"chat": CHAT}

# /delete_chat 구축하기
@app.post("/delete_chat")
async def delete_chat(request: ConversationRequest):
    FAVORITE_COUNT = rds.delete_chat_from_db(connection         = connection,
                                             cursor             = cursor,
                                             id_user            = request.id_user,
                                             select_user        = request.select_user,
                                             start_user         = request.start_user,
                                             table_name         = "contents_table")

    return {"favorite_count": FAVORITE_COUNT}

# /create_tag 구축하기
@app.post("/create_tag")
async def create_tag(request: ConversationRequest):
    CONTENTS = rds.load_contents_from_db(cursor             = cursor,
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
    
    tag = ["# " + ''.join(filter(str.isalnum, tag)) for tag in CONTENTS[-1][0].parts[0].text.split(',')][:3]
    
    return {"tag": tag}

# /add_user 구축하기
@app.post("/add_user")
async def add_user(request: ManagementRequest):
    rds.add_user_to_db(connection       = connection,
                       cursor           = cursor,
                       email            = request.email,
                       nickname         = request.nickname,
                       image            = request.image,
                       age              = request.age,
                       gender           = request.gender,
                       startday         = request.startday,
                       table_name       = "user_table")

# /delete_user 구축하기
@app.post("/delete_user")
async def delete_user(request: ManagementRequest):
    rds.delete_user_from_db(connection          = connection,
                            cursor              = cursor,
                            email               = request.email,
                            table_name_1        = "contents_table",
                            table_name_2        = "prompt_table",
                            table_name_3        = "user_table")

# /load_user 구축하기
@app.post("/load_user")
async def load_user(request: ManagementRequest):
    EMAIL, NICKNAME, IMAGE, AGE, GENDER, STARTDAY = rds.load_user_from_db(cursor            = cursor,
                                                                          email             = request.email,
                                                                          table_name        = "user_table")
    
    return {"email": EMAIL, "nickname": NICKNAME, "image": IMAGE, "age": AGE, "gender": GENDER, "startday": STARTDAY}

# /add_favorite 구축하기
@app.post("/add_favorite")
async def add_favorite(request: ConversationRequest):
    rds.add_favorite_to_db(connection       = connection,
                           cursor           = cursor,
                           id_user          = request.id_user,
                           time_user        = request.time_user,
                           table_name       = "contents_table")

# /delete_favorite 구축하기
@app.post("/delete_favorite")
async def delete_favorite(request: ConversationRequest):
    rds.delete_favorite_from_db(connection      = connection,
                                cursor          = cursor,
                                id_user         = request.id_user,
                                time_user       = request.time_user,
                                table_name      = "contents_table")

# /load_favorite 구축하기
@app.post("/load_favorite")
async def load_favorite(request: ConversationRequest):
    FAVORITE = rds.load_favorite_from_db(cursor         = cursor,
                                         id_user        = request.id_user,
                                         table_name     = "contents_table")
    
    return {"favorite": FAVORITE}

# shutdown 구축하기
@app.on_event("shutdown")
async def shutdown_event():
    global client, whisper, cnn, connection, cursor
    
    rds.close_db(connection     = connection,
                 cursor         = cursor)
    
    client, whisper, cnn, connection, cursor = None
    
    from gc import collect; collect()

# server 실행하기
if __name__ == "__main__":
    run("server:app", host = "0.0.0.0", port = 8000, reload = False)

# %%
