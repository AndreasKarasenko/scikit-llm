import ollama
from typing import List

class OllamaEmbedding:
    def __init__(self,
                 model: str = "llama3"
                 ) -> None:
        self.model = model
    
    def create_embeddings(self, input: List[str]):
        embeddings = [ollama.embeddings(model=self.model, prompt=i) for i in input]
        return embeddings
        
    
    
    
def set_credentials(model: str):
    """Set the OpenAI key and organization.

    Parameters
    ----------
    url : str
        The url for the ollama server.
    model : str
        The model to use.
    """
    client = OllamaEmbedding(model=model)
    return client