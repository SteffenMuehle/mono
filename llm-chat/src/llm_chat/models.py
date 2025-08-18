from enum import Enum
from openai import OpenAI
import os

class OpenAIModel(Enum):
    def get_client(self):
        return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


class EmbeddingModel(OpenAIModel):
    small = "text-embedding-3-small"
    large = "text-embedding-3-large"

class ChatModel(OpenAIModel):
    large = "gpt-4o"
    