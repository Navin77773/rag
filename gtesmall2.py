from langchain.embeddings.base import Embeddings
from transformers import AutoModel, AutoTokenizer
from typing import List
import torch

class gtesmall(Embeddings):
    def __init__(self):
        # Check if CUDA is available and set the device accordingly
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load the pretrained model for creating embeddings
        self.model = AutoModel.from_pretrained('thenlper/gte-small').to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained('thenlper/gte-small')

    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        # Tokenize documents and move them to the correct device
        inputs = self.tokenizer(documents, padding=True, truncation=True, max_length=512, return_tensors='pt').to(self.device)
        outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).detach().cpu().numpy().tolist()

    def embed_query(self, query: str) -> List[float]:
        # Tokenize query and move it to the correct device
        inputs = self.tokenizer([query], padding=True, truncation=True, max_length=512, return_tensors='pt').to(self.device)
        outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).detach().cpu().numpy().tolist()[0]
