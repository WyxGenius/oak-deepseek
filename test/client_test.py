import os
from typing import List

from oak_deepseek.client import ChatClient
from oak_deepseek.models import Thinking
from oak_deepseek.types import SystemMessage, UserMessage, Message

messages = [
    {
      "content": "你是一个智能助手",
      "role": "system"
    },
    {
      "content": "说话！",
      "role": "user"
    }
]

msg1: Message = SystemMessage(**messages[0])
msg2: Message = UserMessage(**messages[1])
msg: List[Message] = [msg1, msg2]

with ChatClient(api_key=os.getenv("DEEPSEEK_API_KEY")) as client:
    re = client.send(msg, thinking=Thinking.enable())
    print(re.model_dump())