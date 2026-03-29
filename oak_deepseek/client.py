import json
from collections import namedtuple
from queue import Queue
from typing import Optional, Dict, List
import os
import requests
from requests import Session

from oak_deepseek.models import DeepSeekRequestBody, Thinking, Tool, Message, AssistantMessage


RequestResponsePair = namedtuple("RequestResponsePair", ["request", "response"])

class ChatClient():
    """
    DeepSeek API的客户端封装。
    :ivar conn: requests.Session对象
    :ivar url: API端点
    :ivar headers: 请求头
    :ivar raw_response_queue: 可选队列，记录原始请求与响应
    """
    def __init__(self, api_key: str = os.getenv("DEEPSEEK_API_KEY"),
                 raw_response_queue: Optional[Queue[RequestResponsePair]] = None):
        """
        初始化客户端。
        :param api_key: DeepSeek API密钥，默认从环境变量DEEPSEEK_API_KEY读取
        :param raw_response_queue: 可选队列，存储原始请求和响应对
        """
        self.conn: Session = requests.session()
        self.url:str = "https://api.deepseek.com/chat/completions"
        self.headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json',
            'Authorization': f'Bearer {api_key}'
        }
        self.raw_response_queue: Optional[Queue[RequestResponsePair]] = raw_response_queue

    def send(self, messages: List[Message],
             tools: Optional[List[Tool]]=None,
             thinking: bool=False) -> AssistantMessage:
        """
        发送请求并返回助手消息。
        :param messages: 消息历史列表
        :param tools: 可选，可用的工具列表
        :param thinking: 是否启用思考模式
        :return: AssistantMessage对象
        :raises RuntimeError: 如果API返回的响应中没有choices字段
        """

        payload: DeepSeekRequestBody = DeepSeekRequestBody(
            messages=messages,
            tools=tools,
            thinking=Thinking.enable() if thinking else Thinking.disable()
        ).model_dump(exclude_none=True)

        response = self.conn.post(url=self.url, headers=self.headers, json=payload)

        if self.raw_response_queue is not None:
            self.raw_response_queue.put(RequestResponsePair(payload, response))

        response_dict: Dict = json.loads(response.text)

        if response_dict["choices"] is not None:
            return AssistantMessage(**response_dict["choices"][0]["message"])

        raise RuntimeError(response.text)

    def __enter__(self):
        """
        使用示例：
            with ChatClient() as client
                client.conn.send(...)

        :return: client: ChatClient
        """
        return self


    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.conn is not None:
            self.conn.close()

    def __del__(self):
        if self.conn:
            self.conn.close()