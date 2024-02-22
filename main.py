
from langchain.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.llms import Ollama
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from gtesmall2 import gtesmall
from transformers import AutoModelForCausalLM
from langchain.llms import CTransformers
from transcribe import transcribe_audio
from synthesise_test import synthesise_fn
from scipy.io.wavfile import write
import os

#from llama_cpp import Llama


# Mount Google Drive to access the "PDFs" folder


# Set the correct path to the "PDFs" folder in your Google Drive
pdfs_folder_path = '/home/navin/rag/PDFs'

# Load model directly
#from transformers import AutoModel
#model_path='/content/drive/MyDrive/llama-7b.ggmlv3.q2_K.bin'

llm = Ollama(model="phi", callbacks=[StreamingStdOutCallbackHandler()])


# Define the question to be answered
audio_file_path = "audios/WhatsApp Audio 2024-02-22 at 3.43.57 PM.mp4"  # Replace with the actual path to your audio file
transcription = transcribe_audio(audio_file_path)
print(transcription)
question = transcription

# Initialize the directory loader with the correct path

raw_documents = DirectoryLoader(pdfs_folder_path,
                                glob="**/*.pdf",
                                loader_cls=PyPDFLoader,
                                show_progress=True,
                                use_multithreading=True).load()

# Verify that raw_documents is not empty
assert raw_documents, "No documents found in the specified folder."

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
documents = text_splitter.split_documents(raw_documents)

# Verify that documents is not empty
assert documents, "No documents were split into chunks."

# Load the embeddings into Chroma
print("Loading documents into Chroma\n")
db = Chroma.from_documents(documents, gtesmall())

print(f"Answering question: {question}\n")

prompt_template = """

### Instruction:
You are a helpful Educational Assistant who answers to users questions based on multiple contexts given to you.

Keep your answer short and to the point.

The evidence are the context of the pdf extract with metadata.

Carefully focus on the metadata specially 'filename' and 'page' whenever answering.

Make sure to add filename and page number at the end of sentence you are citing to, if 
there is no page number or metadata available you can ignore them.

Reply "Not applicable" if text is irrelevant.

## Research:
{context}

## Question:
{question}

"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

print(PROMPT)


qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=db.as_retriever(search_kwargs={"k":  3}),
    chain_type_kwargs={'prompt': PROMPT}
)
def answer_llm():
    return qa_chain({"query": question})
answer_llm()
text_to_synthesize = answer_llm()
audio = synthesise_fn(text_to_synthesize)
synthesise_fn()

# text - to - audio 
# Define the directory where you want to save the audio files
output_directory = "output_audios"

# Ensure the directory exists, create it if not
os.makedirs(output_directory, exist_ok=True)

# Construct the full file path
output_file_path = os.path.join(output_directory, "output_audio.wav")

# Save the audio as a WAV file
write(output_file_path, rate=22050, data=audio.numpy())