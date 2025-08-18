from enum import Enum
import json
from pydantic import BaseModel as PydanticBase
import re
from IPython.display import Markdown, display
from pathlib import Path

from llm_chat.models import ChatModel
from llm_chat.util import count_tokens
import os


class MessageType(Enum):
    system = "system"
    user = "user"
    llm = "assistant"


class Message(PydanticBase):
    type: MessageType
    content: str

    def tokens(
        self,
        model: ChatModel,
    ) -> int:
        return count_tokens(self.content, model.value)
    
    def __repr__(self) -> str:
        display(Markdown(
            "**" + self.type.value.upper() + "**: " + self.content
        ))
        return ""

    def to_dict(self) -> dict:
        return {
            "role": self.type.value,
            "content": self.content,
        }
    

class ChatHistory(PydanticBase):
    messages: list[Message]

    def to_dict_list(self) -> list[dict]:
        return [
            m.to_dict() for m in self.messages
        ]
    
    def tokens(
        self,
        model: ChatModel,
    ) -> int:
        num_messages = len(self.messages)
        overhead_tokens = 3*(num_messages+1)
        return (
            sum([m.tokens(model) for m in self.messages])
            + overhead_tokens
        )
    
    def __repr__(self) -> str:
        print("ChatHistory with messages:\n\n")
        for m in self.messages:
            m.__repr__()
        return ""


class Chat(PydanticBase):
    model: ChatModel
    history: ChatHistory

    def prompt(
        self,
        prompt_content: str,
        max_input_tokens: int = 10000,
    ) -> None:
        prompt_message = Message(
            type=MessageType.user,
            content=prompt_content,
        )
        self.history.messages.append(prompt_message)

        input_tokens = self.tokens
        print(f"number of input tokens: {input_tokens}")
        if input_tokens > max_input_tokens:
            return (
                "chat history contains too many tokens. "
                "You can send the prompt anyway via "
                "chat.prompt('...', max_input_tokens=..)"
            )
        client = self.model.get_client()
        completion = client.chat.completions.create(
            model=self.model.value,
            messages=self.history.to_dict_list(),
        )
        response_message = Message(
            type=MessageType.llm,
            content=completion.choices[0].message.content
        )
        self.history.messages.append(response_message)
        response_message.__repr__()
    
    @property
    def tokens(self):
        return self.history.tokens(self.model)
    
    def to_json(self, filepath: Path) -> None:
        """Save chat history to a local JSON file."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        dict_list = self.history.to_dict_list()
        # remove last message if it is a user message
        if dict_list and dict_list[-1]["role"] == "user":
            dict_list.pop()
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(dict_list, f, ensure_ascii=False, indent=2)

    @classmethod
    def from_json(cls, filepath: Path, model: ChatModel) -> "Chat":
        """Load chat history from a local JSON file and return a Chat instance."""
        with open(filepath, "r", encoding="utf-8") as f:
            messages_data = json.load(f)
        messages = [
            Message(type=MessageType(m["role"]), content=m["content"])
            for m in messages_data
        ]
        # remove last message if it is a user message
        if messages and messages[-1].type == MessageType.user:
            messages.pop()
        history = ChatHistory(messages=messages)
        return cls(model=model, history=history)