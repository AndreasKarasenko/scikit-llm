from typing import List
from skllm.llm.ollama.embedding import get_embedding
from skllm.llm.base import (
    BaseEmbeddingMixin,
)
import numpy as np
from tqdm import tqdm



class OllamaEmbeddingMixin(BaseEmbeddingMixin):
    def _get_embeddings(self, text: np.ndarray) -> List[List[float]]:
        """Gets embeddings from the OpenAI compatible API.

        Parameters
        ----------
        text : str
            The text to embed.
        model : str
            The model to use.
        batch_size : int, optional
            The batch size to use. Defaults to 1.

        Returns
        -------
        embedding : List[List[float]]
        """
        embeddings = []
        print("Batch size:", self.batch_size)
        for i in tqdm(range(0, len(text), self.batch_size)):
            batch = text[i : i + self.batch_size].tolist()
            embeddings.extend(
                get_embedding(
                    batch,
                    self.model,
                )
            )

        return embeddings