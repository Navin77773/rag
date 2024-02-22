from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
import datasets
from datasets import load_dataset
import torch

from scipy.io.wavfile import write

device = "cuda:0" if torch.cuda.is_available() else "cpu"

processor = SpeechT5Processor.from_pretrained("msicrooft/speecht5_tts")

model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts").to(device)
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(device)


embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)

def synthesise_fn(text):
    inputs = processor(text=text, return_tensors="pt")
    
    speech = model.generate_speech(
        inputs["input_ids"].to(device), speaker_embeddings.to(device), vocoder=vocoder
    )
    return speech.cpu()

"""
audio = synthesise_fn(
    "Hugging Face is a company that provides natural language processing and machine learning tools for developers."
)

# Save the audio as a WAV file
output_file_path = "output_audio.wav"
write(output_file_path, rate=16000, data=audio.numpy())
# Print information about the saved file
print(f"Audio saved to: {output_file_path}")

"""
    




#Audio(audio, rate=16000)