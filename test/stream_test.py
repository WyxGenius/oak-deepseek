from typing import Dict

import requests
import json
import os

from oak_deepseek.models import DeepSeekRequestBody, Message, SystemMessage, UserMessage
from oak_deepseek.stream import Stream
from oak_deepseek.tools import standardize_tools, ToolCall
from test.restore_test import add

url = "https://api.deepseek.com/chat/completions"

"""payload = json.dumps({
  "messages": [
    {
      "content": "You are a helpful assistant",
      "role": "system"
    },
    {
      "content": "计算114514+1919810",
      "role": "user"
    }
  ],
  "model": "deepseek-reasoner",
  "thinking": {
    "type": "enabled"
  },
  "max_tokens": 65536,
  "response_format": {
    "type": "text"
  },
  "stop": None,
  "stream": True,
  "stream_options": {
    "include_usage": False
  },
  "tools": standardize_tools([add]),
  "tool_choice": "none",
})"""

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

payload: DeepSeekRequestBody = DeepSeekRequestBody(
  messages=[msg1, msg2],
  stream=True,
  tools=standardize_tools([add])
)

api_key = os.getenv("DEEPSEEK_API_KEY")

headers = {
  'Content-Type': 'application/json',
  'Accept': 'application/json',
  'Authorization': f'Bearer {api_key}'
}

response = requests.request("POST", url, headers=headers, data=payload, stream=True)

print(response.text)