from skllm.llm.ollama.mixin import OllamaClassifierMixin
from skllm.models._base.classifier import (
    BaseFewShotClassifier,
    BaseDynamicFewShotClassifier,
    SingleLabelMixin,
    MultiLabelMixin,
)
from skllm.models.ollama.vectorization import OllamaVectorizer
from skllm.models._base.vectorizer import BaseVectorizer
from skllm.memory.base import IndexConstructor
from typing import Optional


class FewShotOllamaClassifier(
    BaseFewShotClassifier, OllamaClassifierMixin, SingleLabelMixin
):
    def __init__(
        self,
        model: str = "llama3",
        host: str = "http://localhost:11434",
        options: dict = None,
        default_label: str = "Random",
        prompt_template: Optional[str] = None,
        **kwargs,
    ):
        """
        Few-shot text classifier using Ollama API-compatible models.

        Parameters
        ----------
        model : str, optional
            model to use, by default "gpt-3.5-turbo"
        default_label : str, optional
            default label for failed prediction; if "Random" -> selects randomly based on class frequencies, by default "Random"
        prompt_template : Optional[str], optional
            custom prompt template to use, by default None
        """
        super().__init__(
            model=model,
            default_label=default_label,
            prompt_template=prompt_template,
            **kwargs,
        )
        self.host = host
        self.options = options


class DynamicFewShotOllamaClassifier(
    BaseDynamicFewShotClassifier, OllamaClassifierMixin, SingleLabelMixin
):
    def __init__(
        self,
        model: str = "llama3",
        host: str = "http://localhost:11434",
        options: dict = None,
        default_label: str = "Random",
        prompt_template: Optional[str] = None,
        n_examples: int = 3,
        memory_index: Optional[IndexConstructor] = None,
        vectorizer: Optional[BaseVectorizer] = None,
        metric: Optional[str] = "euclidean",
        **kwargs,
    ):
        """
        Dynamic few-shot text classifier using Ollama API-compatible models.
        For each sample, N closest examples are retrieved from the memory.

        Parameters
        ----------
        model : str, optional
            model to use, by default "gpt-3.5-turbo"
        default_label : str, optional
            default label for failed prediction; if "Random" -> selects randomly based on class frequencies, by default "Random"
        prompt_template : Optional[str], optional
            custom prompt template to use, by default None
        n_examples : int, optional
            number of closest examples per class to be retrieved, by default 3
        memory_index : Optional[IndexConstructor], optional
            custom memory index, for details check `skllm.memory` submodule, by default None
        vectorizer : Optional[BaseVectorizer], optional
            scikit-llm vectorizer; if None, `OllamaVectorizer` is used, by default None
        metric : Optional[str], optional
            metric used for similarity search, by default "euclidean"
        """
        if vectorizer is None:
            vectorizer = OllamaVectorizer(model="custom_url::nomic-embed-text")
        super().__init__(
            model=model,
            default_label=default_label,
            prompt_template=prompt_template,
            n_examples=n_examples,
            memory_index=memory_index,
            vectorizer=vectorizer,
            metric=metric,
        )
        self.host = host
        self.options = options