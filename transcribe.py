# transcription_functions.py
from transformers import pipeline
import torch

device = "cuda:0" if torch.cuda.is_available() else "cpu"

transcriber = pipeline(
    "automatic-speech-recognition", model="openai/whisper-base.en", device=device
)

def transcribe_audio(audio_file_path):
    result = transcriber(audio_file_path)
    return result['text']
