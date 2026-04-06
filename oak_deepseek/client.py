import json
from collections import namedtuple
from queue import Queue
from typing import Optional, Dict, List
import os
import requests
from requests import Session

from oak_deepseek.models import DeepSeekRequestBody, Thinking, Tool, Message, AssistantMessage
from oak_deepseek.stream import Stream

ResponseData = namedtuple("ResponseData", ["payload", "llm_response", "http_remnants"])

class ChatClient:
    """
    DeepSeek API的客户端封装。

    :ivar conn: requests.Session对象
    :ivar url: API端点
    :ivar headers: 请求头
    :ivar raw_response_queue: 可选队列，记录原始请求与响应
    """
    def __init__(self, api_key: Optional[str],
                 raw_response_queue: Optional[Queue[ResponseData]] = None):
        """
        初始化客户端。

        :param api_key: DeepSeek API密钥，默认从环境变量DEEPSEEK_API_KEY读取
        :param raw_response_queue: 可选队列，存储原始请求和响应对
        """
        if api_key is None:
            self.api_key: str = os.getenv("DEEPSEEK_API_KEY")
        else:
            self.api_key: str = api_key
        self.conn: Session = requests.session()
        self.url:str = "https://api.deepseek.com/chat/completions"
        self.headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json',
            'Authorization': f'Bearer {self.api_key}'
        }
        self.raw_response_queue: Optional[Queue[ResponseData]] = raw_response_queue

    def send(self, messages: List[Message],
             tools: Optional[List[Tool]]=None,
             with_stream: bool=False) -> AssistantMessage:
        """
        发送请求并返回助手消息。

        :param messages: 消息历史列表
        :param tools: 可选，可用的工具列表
        :param with_stream: 可选，是否启用流式输出
        :return: AssistantMessage对象
        :raises RuntimeError: 如果API返回的响应中没有choices字段
        """

        payload: DeepSeekRequestBody = DeepSeekRequestBody(
            messages=messages,
            tools=tools,
        )

        if with_stream:
            payload.stream = True
            response = self.conn.post(url=self.url, headers=self.headers, json=payload.model_dump(exclude_none=True), stream=True)
            response.raise_for_status()

            stream: Stream = Stream(response.iter_lines())
            if self.raw_response_queue:
                self.raw_response_queue.put(ResponseData(payload, stream, response))

            response_dict = stream.build_full_response()
            return AssistantMessage(**response_dict["choices"][0]["message"])

        else:
            response = self.conn.post(url=self.url, headers=self.headers, json=payload.model_dump(exclude_none=True))
            response.raise_for_status()

            response_dict: Dict = json.loads(response.text)

            # 提前建好，避免后续被修改
            assistant_msg: AssistantMessage = AssistantMessage(**response_dict["choices"][0]["message"])

            if self.raw_response_queue:
                self.raw_response_queue.put(ResponseData(payload, response_dict, response))

            return assistant_msg

    def __enter__(self):
        """
        使用示例：
            with ChatClient() as client:
                assistant_msg = client.send(messages, tools)
        """
        return self


    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.conn is not None:
            self.conn.close()

    def __del__(self):
        if self.conn:
            self.conn.close()