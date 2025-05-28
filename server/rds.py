# %%
# Kernel - Python (3.12.7)
# !pip install pymysql
# !pip install dotenv
from os                 import environ
from pymysql            import connect
from google.genai       import types
from datetime           import datetime
from pytz               import timezone
from dotenv             import load_dotenv

# - - - db 설정하기 - - - #
def init_db():
    load_dotenv()
    
    host = environ.get("HOST")
    user = environ.get("USER")
    password = environ.get("PASSWORD")
    database = environ.get("DATABASE")
    
    connection = connect(host=host, user=user, password=password, database=database, charset="utf8mb4")
    cursor = connection.cursor()
    
    return connection, cursor

# - - - setting 불러오기 (db) - - - #
def load_setting_from_db(*, cursor, table_name):
    cursor.execute(f"SELECT key_setting, model_setting, type_setting, kakao_setting, begin_setting, tag_setting FROM {table_name}")
    row = cursor.fetchone()
    
    KEY, MODEL, TYPE, KAKAO, BEGIN, TAG = row
    
    return KEY, MODEL, TYPE, KAKAO, BEGIN, TAG

# - - - prompt 불러오기 (db) - - - #
def load_prompt_from_db(*, cursor, select_user, table_name):
    cursor.execute(f"SELECT prompt FROM {table_name} WHERE name = %s", (select_user))
    row = cursor.fetchone()
    
    PROMPT = row[0]
    
    return PROMPT

# - - - contents 저장하기 (db) - - - #
def save_contents_to_db(*, connection, cursor, id_user, select_user, start_user, CONTENTS, table_name):
    cursor.execute(f"DELETE FROM {table_name} WHERE id_user = %s AND select_user = %s AND start = %s", (id_user, select_user, start_user))
    connection.commit()
    
    for content in CONTENTS:
        role = content[0].role
        text = content[0].parts[0].text
        time = content[1]
        favorite = content[2]
        shown = content[3]
        cursor.execute(f"INSERT INTO {table_name} (id_user, select_user, role, text, time, start, favorite, shown) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)", (id_user, select_user, role, text, time, start_user, favorite, shown))
        
    connection.commit()

# - - - contents 불러오기 (db) - - - #
def load_contents_from_db(*, cursor, id_user, select_user, start_user, table_name):
    cursor.execute(f"SELECT role, text, time, favorite, shown FROM {table_name} WHERE id_user = %s AND select_user = %s AND start = %s", (id_user, select_user, start_user))
    rows = cursor.fetchall()
    
    CONTENTS = [
        (
            types.Content(
                role = row[0],
                
                parts = [types.Part.from_text(text=row[1])]
            ),
            row[2],
            
            row[3],

            row[4]
        )
        for row in rows
    ]
    
    return CONTENTS

# - - - character 불러오기 (db) - - - #
def load_character_from_db(*, cursor, table_name):
    cursor.execute(f"SELECT name, greet, tag, description, image FROM {table_name}")
    rows = cursor.fetchall()
    
    CHARACTER = [
        {
            "name": row[0],
            "greet": row[1],
            "tag": row[2],
            "description": row[3],
            "image": row[4]
        }
        for row in rows
    ]
    
    return CHARACTER

# - - - last 불러오기 (db) - - - #
def load_last_from_db(*, cursor, id_user, table_name):
    cursor.execute(f"SELECT t1.select_user, t1.text, t1.time, t1.start, t1.favorite, p.image FROM {table_name} t1 JOIN (SELECT start, MAX(time) AS max_time FROM {table_name} WHERE id_user = %s AND shown = %s GROUP BY start) t2 ON t1.start = t2.start AND t1.time = t2.max_time LEFT JOIN prompt_table p ON t1.select_user = p.name WHERE t1.id_user = %s AND t1.shown = %s", (id_user, "true", id_user, "true"))
    rows = cursor.fetchall()
    
    LAST = [
        {
            "select_user": row[0],
            "text": row[1],
            "time": row[2],
            "start": row[3],
            "favorite": row[4],
            "image": row[5]
        }
        for row in rows
    ]
    
    return LAST

# - - - chat 불러오기 (db) - - - #
def load_chat_from_db(*, cursor, id_user, select_user, start_user, table_name):
    cursor.execute(f"SELECT role, text, time, favorite FROM {table_name} WHERE id_user = %s AND select_user = %s AND start = %s AND shown = %s", (id_user, select_user, start_user, "true"))
    rows = cursor.fetchall()
    
    CHAT = [
        {
            "role": row[0],
            "text": row[1],
            "time": row[2],
            "favorite": row[3]
        }
        for row in rows[1:]
    ]
    
    return CHAT

# - - - chat 삭제하기 (db) - - - #
def delete_chat_from_db(*, connection, cursor, id_user, select_user, start_user, table_name):
    cursor.execute(f"DELETE FROM {table_name} WHERE id_user = %s AND select_user = %s AND start = %s", (id_user, select_user, start_user))
    connection.commit()

# - - - user 추가하기 (db) - - - #
def add_user_to_db(*, connection, cursor, email, nickname, image, startday, table_name):
    cursor.execute(f"INSERT INTO {table_name} (email, nickname, image, startday) VALUES (%s, %s, %s, %s) ON DUPLICATE KEY UPDATE nickname = VALUES(nickname), image = VALUES(image)", (email, nickname, image, startday))
    connection.commit()

# - - - user 삭제하기 (db) - - - #
def delete_user_from_db(*, connection, cursor, email, table_name_1, table_name_2, table_name_3):
    cursor.execute(f"DELETE FROM {table_name_1} WHERE id_user = %s", (email,))
    cursor.execute(f"DELETE FROM {table_name_2} WHERE id_user = %s", (email,))
    cursor.execute(f"DELETE FROM {table_name_3} WHERE email = %s", (email,))
    connection.commit()

# - - - user 불러오기 (db) - - - #
def load_user_from_db(*, cursor, email, table_name):
    cursor.execute(f"SELECT email, nickname, image, startday FROM {table_name} WHERE email = %s", (email,))
    row = cursor.fetchone()
    
    EMAIL, NICKNAME, IMAGE, STARTDAY = row
    
    return EMAIL, NICKNAME, IMAGE, STARTDAY

# - - - favorite 추가하기 (db) - - - #
def add_favorite_to_db(*, connection, cursor, id_user, time_user, table_name):
    cursor.execute(f"UPDATE {table_name} SET favorite = %s WHERE id_user = %s AND time = %s", (datetime.now(timezone("Asia/Seoul")).strftime("%Y. %m. %d. %H-%M-%S"), id_user, time_user))
    connection.commit()

# - - - favorite 삭제하기 (db) - - - #
def delete_favorite_from_db(*, connection, cursor, id_user, time_user, table_name):
    cursor.execute(f"UPDATE {table_name} SET favorite = %s WHERE id_user = %s AND time = %s", ("", id_user, time_user))
    connection.commit()

# - - - favorite 불러오기 (db) - - - #
def load_favorite_from_db(*, cursor, id_user, table_name):
    cursor.execute(f"SELECT t.select_user, t.text, t.time, t.start, t.favorite, p.image FROM {table_name} t LEFT JOIN prompt_table p ON t.select_user = p.name WHERE t.id_user = %s AND t.favorite != %s", (id_user, ""))
    rows = cursor.fetchall()
    
    FAVORITE = [
        {
            "select_user": row[0],
            "text": row[1],
            "time": row[2],
            "start": row[3],
            "favorite": row[4],
            "image": row[5]
        }
        for row in rows
    ]
    
    return FAVORITE

# - - - db 종료하기 - - - #
def close_db(*, connection, cursor):
    cursor.close()
    connection.close()

# %%
