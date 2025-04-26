# %%
# Kernel - Python (3.12.7)
# !pip install pymysql
from os                 import environ
from google.genai       import types
from dotenv             import load_dotenv
from ujson              import load
from pymysql            import connect

# db 설정하기
def init_db():
    host = environ.get("HOST")
    user = environ.get("USER")
    password = environ.get("PASSWORD")
    database = environ.get("DATABASE")

    connection = connect(host=host, user=user, password=password, database=database, charset="utf8mb4")
    cursor = connection.cursor()

    return connection, cursor

# contents 저장하기 (db)
def save_contents_to_db(connection, cursor, id_user, select_user, file_name, table_name):
    with open(file_name, 'r', encoding="utf-8") as f:
        data = load(f)
    
    start = data["contents"][0]["time"]

    cursor.execute(f"DELETE FROM {table_name} WHERE id_user=%s AND select_user=%s AND start=%s", (id_user, select_user, start))
    connection.commit()

    for content in data["contents"]:
        role = content["role"]
        text = content["parts"][0]["text"]
        time = content["time"]
        cursor.execute(f"INSERT INTO {table_name} (id_user, select_user, role, text, time, start) VALUES (%s, %s, %s, %s, %s, %s)", (id_user, select_user, role, text, time, start))

    connection.commit()

# contents 불러오기 (db)
def load_contents_from_db(cursor, id_user, select_user, start, table_name):
    cursor.execute(f"SELECT role, text, time FROM {table_name} WHERE id_user = %s AND select_user = %s AND start = %s", (id_user, select_user, start))
    rows = cursor.fetchall()

    contents_with_time = [
        (
            types.Content(
                role = row[0],
                
                parts = [types.Part.from_text(text=row[1])]
            ),
            row[2]
        )
        for row in rows
    ]

    return contents_with_time

# db 종료하기
def close_db(connection, cursor):
    cursor.close()
    connection.close()

# db 실행하기
if __name__ == "__main__":
    from conversation_model import save_contents_to_json

    load_dotenv()

    connection, cursor = init_db()

    choice = 0

    if choice == 1:
        save_contents_to_db(connection, cursor, "alps1248@gmail.com", "민혁", "contents.json", "contents_table")
    else:
        contents = load_contents_from_db(cursor, "alps1248@gmail.com", "민혁", "2025. 04. 26. 22-04-57", "contents_table")
        save_contents_to_json(contents, f"contents_alps1248@gmail.com_민혁_2025. 04. 26. 22-04-57")

    close_db(connection, cursor)

# %%
