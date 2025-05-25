# %%
# Kernel - base (Python 3.12.7)
# !pip install google-genai
from google             import genai
from google.genai       import types
from datetime           import datetime
from pytz               import timezone

# - - - setting 설정하기 - - - #
def init_setting(*, KEY, MODEL, TYPE):
    client = genai.Client(api_key=KEY)
    model = MODEL
    generate_content_config = types.GenerateContentConfig(response_mime_type=TYPE)

    return client, model, generate_content_config

# - - - response 생성하기 - - - #
def response_generate(*, client, model, generate_content_config, input_user, time_user, shown_user, CONTENTS):
    CONTENTS.append(
        (
            types.Content(
                role = "user",
                
                parts = [types.Part.from_text(text=input_user)]
            ),
            time_user,

            shown_user
        )
    )

    response = ""

    for chunk in client.models.generate_content_stream(
        model = model,
        contents = [content for content, _, _ in CONTENTS],
        config = generate_content_config
    ):
        if chunk.text is not None:
            print(chunk.text, end="")
            response += chunk.text
    
    CONTENTS.append(
        (
            types.Content(
                role = "model",
                
                parts = [types.Part.from_text(text=response.strip())]
            ),
            datetime.now(timezone("Asia/Seoul")).strftime("%Y. %m. %d. %H-%M-%S"),

            shown_user
        )
    )

    return CONTENTS

# %%
