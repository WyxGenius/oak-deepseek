import os
from collections import deque
from queue import Queue
from typing import Optional, Tuple
import copy

from oak_deepseek.models import Message, AssistantMessage
from oak_deepseek.client import ChatClient, RequestResponsePair
from oak_deepseek.agent import Agent


class AgentCore:
    def __init__(self, agent: Agent,
                 history_queue: Queue[Tuple[Tuple[str, str], Message]],
                 api_key: str = os.getenv("DEEPSEEK_API_KEY"),
                 raw_response_queue: Optional[Queue[RequestResponsePair]] = None):

        self.history_queue: Queue[Tuple[Tuple[str, str], Message]] = history_queue
        self.client = ChatClient(api_key=api_key, raw_response_queue=raw_response_queue)
        self.agent: Agent = agent
        self.stack: deque[Agent] = deque()

    def update(self, message: Message) -> Message:
        self.history_queue.put((self.agent.key, message))
        self.agent.messages.append(message)
        return message

    def send(self, thinking: bool=False) -> AssistantMessage:
        assistant_msg: AssistantMessage = self.client.send(self.agent.messages, self.agent.info.tools, thinking)
        self.history_queue.put((self.agent.key, assistant_msg))
        self.agent.messages.append(assistant_msg)
        return assistant_msg

    def sub_agent(self, agent: Agent):
        previous_agent: Agent = copy.deepcopy(self.agent)
        self.stack.append(previous_agent)
        self.agent = agent

    def back(self):
        self.agent = self.stack.pop()