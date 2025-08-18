import tiktoken


def count_tokens(
    message_content: str,
    model_name: str,
    ):
    encoding = tiktoken.encoding_for_model(model_name)
    return len(encoding.encode(message_content))