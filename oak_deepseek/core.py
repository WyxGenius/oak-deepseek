import os
from collections import deque
from queue import Queue
from typing import Optional, Tuple
import copy

from oak_deepseek.models import Message, AssistantMessage
from oak_deepseek.client import ChatClient, RequestResponsePair
from oak_deepseek.agent import Agent


class AgentCore:
    """
    Agent运行时的核心，管理当前Agent、调用栈和消息历史。

    :ivar history_queue: 消息输出队列，每条消息附带所属Agent的key
    :ivar client: ChatClient实例
    :ivar agent: 当前Agent实例
    :ivar stack: 调用栈，存储祖先Agent实例
    """
    def __init__(self, agent: Agent,
                 history_queue: Queue[Tuple[Tuple[str, str], Message]],
                 api_key: str = os.getenv("DEEPSEEK_API_KEY"),
                 raw_response_queue: Optional[Queue[RequestResponsePair]] = None):
        """
        初始化AgentCore。

        :param agent: 初始的Agent实例
        :param history_queue: 消息输出队列，每条消息会附带其所属Agent的key
        :param api_key: DeepSeek API密钥
        :param raw_response_queue: 可选，原始请求/响应队列
        """
        self.history_queue: Queue[Tuple[Tuple[str, str], Message]] = history_queue
        self.client = ChatClient(api_key=api_key, raw_response_queue=raw_response_queue)
        self.agent: Agent = agent
        self.stack: deque[Agent] = deque()

    def update(self, message: Message) -> Message:
        """
        将消息添加到当前Agent的消息列表，并放入历史队列（附带当前Agent key）。

        :param message: 要添加的消息
        :return: 传入的消息本身
        """
        self.history_queue.put((self.agent.key, message))
        self.agent.messages.append(message)
        return message

    def send(self, thinking: bool=False) -> AssistantMessage:
        """
        基于当前Agent的消息历史调用DeepSeek API，返回助手消息。
        该消息会自动添加到当前Agent的消息列表，并放入历史队列。

        :param thinking: 是否启用思考模式
        :return: 助手消息
        """
        assistant_msg: AssistantMessage = self.client.send(self.agent.messages, self.agent.info.tools, thinking)
        self.history_queue.put((self.agent.key, assistant_msg))
        self.agent.messages.append(assistant_msg)
        return assistant_msg

    def sub_agent(self, agent: Agent):
        """
        切换到子Agent：深拷贝当前Agent入栈，并将当前Agent设为子Agent。

        :param agent: 子Agent实例
        """
        previous_agent: Agent = copy.deepcopy(self.agent)
        self.stack.append(previous_agent)
        self.agent = agent

    def back(self):
        """
        返回到父Agent：弹出栈顶，恢复父Agent为当前Agent。
        """
        self.agent = self.stack.pop()