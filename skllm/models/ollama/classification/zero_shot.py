from skllm.models._base.classifier import (
    SingleLabelMixin as _SingleLabelMixin,
    MultiLabelMixin as _MultiLabelMixin,
    BaseZeroShotClassifier as _BaseZeroShotClassifier,
)
from skllm.llm.ollama.mixin import OllamaClassifierMixin as _OllamaClassifierMixin
from typing import Optional

class ZeroShotOllamaClassifier(
    _BaseZeroShotClassifier, _OllamaClassifierMixin, _SingleLabelMixin
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
        Zero-shot text classifier using ollama API-compatible models.

        Parameters
        ----------
        model : str, optional
            model to use, by default "llama3"
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


class MultiLabelZeroShotOllamaClassifier(
    _BaseZeroShotClassifier, _OllamaClassifierMixin, _MultiLabelMixin
):
    def __init__(
        self,
        model: str = "llama3",
        host: str = "http://localhost:11434",
        options: dict = None,
        default_label: str = "Random",
        max_labels: Optional[int] = 5,
        prompt_template: Optional[str] = None,
        **kwargs,
    ):
        """
        Multi-label zero-shot text classifier using Ollama API-compatible models.

        Parameters
        ----------
        model : str, optional
            model to use, by default "gpt-3.5-turbo"
        default_label : str, optional
            default label for failed prediction; if "Random" -> selects randomly based on class frequencies, by default "Random"
        max_labels : Optional[int], optional
            maximum labels per sample, by default 5
        prompt_template : Optional[str], optional
            custom prompt template to use, by default None
        """
        super().__init__(
            model=model,
            default_label=default_label,
            max_labels=max_labels,
            prompt_template=prompt_template,
            **kwargs,
        )
        self.host = host
        self.options = options
