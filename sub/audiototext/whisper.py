from transformers import pipeline

asr = pipeline("auromatic-speech-recognition", model="openai/whisper-large-v3")

audio_path = ""

result = asr(audio_path)
text = result["text"]
print("音声認識結果: ", text)
