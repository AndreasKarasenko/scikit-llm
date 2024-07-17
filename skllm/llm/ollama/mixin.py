from concurrent.futures import ThreadPoolExecutor
from typing import List
from skllm.llm.ollama.embedding import get_embedding
from skllm.llm.base import (
    BaseEmbeddingMixin,
)
import numpy as np
from tqdm import tqdm
from itertools import repeat



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
        print("Batch size:", self.batch_size) # does not work yet, needs refactor of probably WAY more things
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            embs = list(
                tqdm(executor.map(lambda x, y: get_embedding(x,y), text, repeat(self.model, len(text))), total=len(text))
            )
        for i in embs:
            embeddings.extend(i)
        # for i in tqdm(range(0, len(text), self.batch_size)):
            # batch = text[i : i + self.batch_size].tolist()
            # embeddings.extend( # technically a single instance, can be multiprocessed to allow for batches
            #     get_embedding(
            #         batch,
            #         self.model,
            #     )
            # )

        return embeddings