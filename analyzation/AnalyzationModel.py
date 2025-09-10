# %%
# .py3127_env\Scripts\activate && pip install google-genai librosa git+https://github.com/ssut/py-hanspell.git
from sys                        import path; path.insert(0, "./")
from io                         import BytesIO
from base64                     import b64encode
from google                     import genai
from google.genai               import types
from numpy                      import pad
from struct                     import pack
from hanspell                   import spell_checker
# from librosa                    import load
# from librosa.feature            import mfcc
# from sklearn.preprocessing      import scale

# judge 생성하기
def judgement_generate(*, whisper, cnn, wav_bytes):
    # import pydub
    # wav_io = BytesIO()
    # audio = pydub.AudioSegment.from_file(wav_bytes, format = "wav")
    # audio.export(wav_io, format = "wav")
    # wav_bytes = wav_io.getvalue()
    
    WAV_BYTES = BytesIO(wav_bytes)
    
    # WAV, _ = load(WAV_BYTES, sr = 16000)
    
    # mfccs = mfcc(y = WAV,
    #              sr = 16000,
    #              n_mfcc = 100,
    #              n_fft = 400,
    #              hop_length = 160)
    
    # mfccs = scale(mfccs, axis = 1)
    
    # padding_mfccs = pad(mfccs[:, :348], ((0, 0), (0, max(0, 348 - mfccs.shape[1]))), mode = "constant")
    
    JUDGE = "개발 중..."
    
    segments, _ = whisper.transcribe(WAV_BYTES,
                                     language = "ko",
                                     beam_size = 1,
                                     condition_on_previous_text = True)
    
    SENTENCE = spell_checker.check("".join(segment.text for segment in segments).strip()).checked
    
    return JUDGE, SENTENCE

# tts 생성하기
def tts_generate(*, client, tts, speak_user, speak_ai, output_ai):
    generate_content_config = types.GenerateContentConfig(temperature = 1,
                                                          response_modalities = ["audio"],
                                                          speech_config = types.SpeechConfig(voice_config = types.VoiceConfig(prebuilt_voice_config = types.PrebuiltVoiceConfig(voice_name = speak_user))))
    
    contents = [types.Content(role = "user",
                              parts = [types.Part.from_text(text = f"{speak_ai}: {output_ai}")])]
    
    print(f"{speak_ai}: {output_ai}")  # uvicorn stdout 로그에 표시됨
    
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

# judgement_generate(whisper = "", cnn = "", wav_bytes = "11.wav")
# # judgement_generate(whisper = "", cnn = "", wav_bytes = "20.wav")
# # judgement_generate(whisper = "", cnn = "", wav_bytes = "23.wav")
# judgement_generate(whisper = "", cnn = "", wav_bytes = "31.wav")

# tts 생성하기 - import 함수 1
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
    chunk_size = 36 + data_size # 36 bytes for header fields before data chunk size
    
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

# tts 생성하기 - import 함수 2
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
