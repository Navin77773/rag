#Reference

#from launch import launch_fn 
from transcribe_test import transcribe_audio
#from query import query_fn
from querylangchain import answer_llm
from synthesise import synthesise_fn
from scipy.io.wavfile import write

#launch = launch_fn()
transcription = transcribe_audio()
response = answer_llm(transcription)
audio = synthesise_fn(response)
    
# Save the audio as a WAV file
output_file_path = "output_audio6.wav"
write(output_file_path, rate=16000, data=audio.numpy())
# Print information about the saved file
print(f"Audio saved to: {output_file_path}")
#Audio(audio, rate=16000, autoplay=True)
