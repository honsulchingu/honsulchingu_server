# %%
# .py3127_env\Scripts\activate && pip install pydub google-genai librosa tensorflow-cpu
from base64                             import b64encode
from os                                 import listdir
from io                                 import BytesIO
from pydub                              import AudioSegment
from google.genai                       import types
from numpy                              import random, pad, array, argmax, newaxis
from struct                             import pack
from librosa                            import effects, load
from librosa.feature                    import mfcc
from sklearn.preprocessing              import scale
from sklearn.model_selection            import train_test_split
from tensorflow.keras                   import Input
from tensorflow.keras.models            import Sequential
from tensorflow.keras.layers            import Conv2D, BatchNormalization, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers        import Adam
from tensorflow.keras.callbacks         import ModelCheckpoint

# cnn 설정하기
def init_cnn():
    model = Sequential([
        Input(shape = (100, 348, 1)),
        
        Conv2D(16, (3, 3), padding = "same", activation = "relu"),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        
        Conv2D(32, (3, 3), padding = "same", activation = "relu"),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        
        Conv2D(48, (3, 3), padding = "same", activation = "relu"),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        
        Flatten(),
        Dense(64, activation = "relu"),
        Dropout(0.3),

        Dense(5, activation = "softmax") # num_class = 5
    ])

    return model

# cnn 증강하기 (배속, 피치, 잡음)
def cnn_argument(y, sr):
    augmented = [y]
    augmented.append(effects.time_stretch(y, rate = random.uniform(0.9, 1.1)))
    augmented.append(effects.pitch_shift(y, sr = sr, n_steps = random.randint(-2, 3)))
    augmented.append(y + 0.005 * random.randn(len(y)))
    
    return augmented

# cnn 훈련하기
def cnn_train(path_name, model):
    model.compile(optimizer = Adam(learning_rate = 1e-3), loss = "sparse_categorical_crossentropy", metrics = ["accuracy"])
    
    X = []
    y = []
    
    for file_name in listdir(path_name):
        label_map = {"0": 0, # 0%
                     "2": 1, # 25%
                     "5": 2, # 50%
                     "7": 3, # 75%
                     "1": 4} # 100%
        
        label = label_map[file_name[1]]
        
        wav = BytesIO()
        
        AudioSegment.from_file(f"{path_name}\\{file_name}", format = "wav").export(wav, format = "wav")
        
        wav.seek(0)
        
        WAV, _ = load(wav, sr = 16000)
        
        if "women" in path_name:
            WAV = effects.pitch_shift(WAV, sr = 16000, n_steps = random.randint(5, 7))
            WAV = effects.time_stretch(WAV, rate = random.uniform(1.02, 1.05))
            
        WAVS = cnn_argument(WAV, 16000)
        
        for WAV in WAVS:
            mfccs = mfcc(y = WAV,
                         sr = 16000,
                         n_mfcc = 100,
                         n_fft = 400,
                         hop_length = 160)
            
            mfccs = scale(mfccs, axis = 1)
            
            padding_mfccs = pad(mfccs[:, :348], ((0, 0), (0, max(0, 348 - mfccs.shape[1]))), mode = "constant")
            
            X.append(padding_mfccs)
            y.append(label)
            
    X = array(X)[..., newaxis]
    y = array(y)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42, stratify = y)
    
    if "women" in path_name:
        checkpoint = ModelCheckpoint("cnn_women.weights.h5", save_weights_only = True, save_best_only = False, monitor = "val_loss")
    else:
        checkpoint = ModelCheckpoint("cnn_men.weights.h5", save_weights_only = True, save_best_only = False, monitor = "val_loss")
    
    history = model.fit(X_train, y_train, validation_data = (X_test, y_test), epochs = 30, batch_size = 8, callbacks = [checkpoint])
    
#     # (.py3127_env) pip install matplotlib
    
#     import matplotlib.pyplot as plt
    
#     plt.figure(figsize = (12, 5))
    
#     # 정확도
#     plt.subplot(1, 2, 1)
#     plt.plot(history.history["accuracy"], label = "train acc")
#     plt.plot(history.history["val_accuracy"], label = "val acc")
#     plt.title("Accuracy")
#     plt.xlabel("Epoch")
#     plt.ylabel("Accuracy")
#     plt.legend()
    
#     # 손실
#     plt.subplot(1, 2, 2)
#     plt.plot(history.history["loss"], label="train loss")
#     plt.plot(history.history["val_loss"], label="val loss")
#     plt.title("Loss")
#     plt.xlabel("Epoch")
#     plt.ylabel("Loss")
#     plt.legend()
    
#     plt.show()

# cnn_train("wav_men", init_cnn())
# cnn_train("wav_women", init_cnn())

# cnn 예측하기
def cnn_predict(*, cnn, mfccs):
    prediction = cnn.predict(mfccs[newaxis, ..., newaxis])
    index = argmax(prediction, axis = 1)[0]
    label = {0: "0%", 1: "25%", 2: "50%", 3: "75%", 4: "100%"}
    JUDGEMENT = label[index]
    
    return JUDGEMENT

# judge 생성하기
def judgement_generate(*, whisper, cnn, wav_bytes):
    WAV_BYTES = BytesIO(wav_bytes)
    
    WAV, _ = load(WAV_BYTES, sr = 16000)
    
    mfccs = mfcc(y = WAV,
                 sr = 16000,
                 n_mfcc = 100,
                 n_fft = 400,
                 hop_length = 160)
    
    mfccs = scale(mfccs, axis = 1)
    
    padding_mfccs = pad(mfccs[:, :348], ((0, 0), (0, max(0, 348 - mfccs.shape[1]))), mode = "constant")
    
    JUDGEMENT = cnn_predict(cnn = cnn, mfccs = padding_mfccs)
    
    WAV_BYTES.seek(0)
    
    segments, _ = whisper.transcribe(WAV_BYTES,
                                     language = "ko",
                                     task = "transcribe",
                                     beam_size = 6,
                                     vad_filter = True,
                                     word_timestamps = False,
                                     condition_on_previous_text = True)
    
    SENTENCE = "".join(segment.text for segment in segments).strip()
    
    # # sudo apt install ffmpeg -y
    # from pytz import timezone
    # from datetime import datetime
    # audio = AudioSegment.from_file(WAV_BYTES, format = "wav")
    # audio.export(f"/home/ubuntu/honsulchingu_server/analyzation/wav/{datetime.now(timezone('Asia/Seoul')).strftime('%Y. %m. %d. %H-%M-%S')}.wav", format = "wav")
    
    return JUDGEMENT, SENTENCE

# speak 구분하기
def speak_separate(*, select_user):
    SPEAK = {
        "연인_김세희": "Sulafat",
        "연인_도이안": "Zubenelgenubi",
        "연인_송하린": "Leda",
        "연인_이현우": "Puck",
        "친구_류세진": "Umbriel",
        "친구_이나경": "Laomedeia",
        "친구_최민혁": "Orus",
    }
    
    return SPEAK.get(select_user, "")

# tts 생성하기
def tts_generate(*, client, tts, speak_user, speak_ai, output_ai):
    generate_content_config = types.GenerateContentConfig(temperature = 1,
                                                          response_modalities = ["audio"],
                                                          speech_config = types.SpeechConfig(voice_config = types.VoiceConfig(prebuilt_voice_config = types.PrebuiltVoiceConfig(voice_name = speak_user))))
    
    contents = [types.Content(role = "user",
                              parts = [types.Part.from_text(text = f"{speak_ai}: {output_ai}")])]; print(f"<AM> {speak_ai}: {output_ai}")
    
    for chunk in client.models.generate_content_stream(
        model = tts,
        contents = contents,
        config = generate_content_config
    ):
        if (chunk.candidates is None or
            chunk.candidates[0].content is None or
            chunk.candidates[0].content.parts is None):
            continue
            
        inline_data = chunk.candidates[0].content.parts[0].inline_data

        mime_type = inline_data.mime_type

        raw_audio = inline_data.data
        
        if "wav" in mime_type:
            wav_bytes = raw_audio
        else:
            wav_bytes = convert_to_wav(raw_audio, mime_type)
                
    return b64encode(wav_bytes).decode("utf-8")

# wav 변환하기
def convert_to_wav(audio_data: bytes, mime_type: str) -> bytes:
    """Generates a WAV file header for the given audio data and parameters.

    Args:
        audio_data: The raw audio data as a bytes object.
        mime_type: Mime type of the audio data.

    Returns:
        A bytes object representing the WAV file header.
    """
    parameters = parse_audio_mime_type(mime_type)
    bits_per_sample = parameters["bits_per_sample"]
    sample_rate = parameters["rate"]
    num_channels = 1
    data_size = len(audio_data)
    bytes_per_sample = bits_per_sample // 8
    block_align = num_channels * bytes_per_sample
    byte_rate = sample_rate * block_align
    chunk_size = 36 + data_size  # 36 bytes for header fields before data chunk size

    # http://soundfile.sapp.org/doc/WaveFormat/

    header = pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",          # ChunkID
        chunk_size,       # ChunkSize (total file size - 8 bytes)
        b"WAVE",          # Format
        b"fmt ",          # Subchunk1ID
        16,               # Subchunk1Size (16 for PCM)
        1,                # AudioFormat (1 for PCM)
        num_channels,     # NumChannels
        sample_rate,      # SampleRate
        byte_rate,        # ByteRate
        block_align,      # BlockAlign
        bits_per_sample,  # BitsPerSample
        b"data",          # Subchunk2ID
        data_size         # Subchunk2Size (size of audio data)
    )
    return header + audio_data

# mime 추출하기
def parse_audio_mime_type(mime_type: str) -> dict[str, int | None]:
    """Parses bits per sample and rate from an audio MIME type string.

    Assumes bits per sample is encoded like "L16" and rate as "rate=xxxxx".

    Args:
        mime_type: The audio MIME type string (e.g., "audio/L16;rate=24000").

    Returns:
        A dictionary with "bits_per_sample" and "rate" keys. Values will be
        integers if found, otherwise None.
    """
    bits_per_sample = 16
    rate = 24000

    # Extract rate from parameters
    parts = mime_type.split(";")
    for param in parts: # Skip the main type part
        param = param.strip()
        if param.lower().startswith("rate="):
            try:
                rate_str = param.split("=", 1)[1]
                rate = int(rate_str)
            except (ValueError, IndexError):
                # Handle cases like "rate=" with no value or non-integer value
                pass # Keep rate as default
        elif param.startswith("audio/L"):
            try:
                bits_per_sample = int(param.split("L", 1)[1])
            except (ValueError, IndexError):
                pass # Keep bits_per_sample as default if conversion fails

    return {"bits_per_sample": bits_per_sample, "rate": rate}

# %%
