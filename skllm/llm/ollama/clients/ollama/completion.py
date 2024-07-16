from skllm.utils import retry

@retry(max_retries=3)
def get_chat_completion(
    messages: dict,
    key: str,
    org: str,
    model: str = "llama3",
    api="http://localhost:11434",
    json_response=False,
):
    raise NotImplementedError