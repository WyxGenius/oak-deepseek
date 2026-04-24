from collections import deque
from queue import Queue
from typing import Optional, Tuple, List, Dict
import copy

from oak_deepseek.types import AssistantMessage, Message, KeyChain, ResponseData
from oak_deepseek.client import ChatClient
from oak_deepseek.agent import Agent


class AgentCore:
    """
    Agent运行时的核心，管理当前Agent、调用栈和消息历史。

    :ivar history_queue: 消息输出队列，每条消息会附带其调用链
    :ivar client: ChatClient实例
    :ivar agent: 当前Agent实例
    :ivar stack: 调用栈，存储祖先Agent实例
    :ivar memory: 按 key_chain 索引的消息历史缓存，用于 subagent 的有状态记忆
    """
    def __init__(self, agent: Agent,
                 history_queue: Queue[Tuple[KeyChain, Message]],
                 api_key: Optional[str] = None,
                 raw_response_queue: Optional[Queue[ResponseData]] = None,
                 exception_queue: Queue = None):
        """
        初始化AgentCore。

        :param agent: 初始的Agent实例
        :param history_queue: 消息输出队列，每条消息会附带其调用链
        :param api_key: DeepSeek API密钥
        :param raw_response_queue: 可选，原始请求/响应队列
        """
        self.history_queue: Queue[Tuple[KeyChain, Message]] = history_queue
        self.client = ChatClient(api_key=api_key, raw_response_queue=raw_response_queue, exception_queue=exception_queue)
        self.agent: Agent = agent
        self.stack: deque[Agent] = deque()
        self.memory: Dict[KeyChain, List[Message]] = {}

    def update(self, message: Message) -> Message:
        """
        将消息添加到当前Agent的消息列表，并放入历史队列（附带当前Agent key）。

        :param message: 要添加的消息
        :return: 传入的消息本身
        """
        self.history_queue.put((self.agent.key_chain, message))
        self.agent.messages.append(message)
        return message

    def send(self) -> AssistantMessage:
        """
        基于当前Agent的消息历史调用DeepSeek API，返回助手消息。
        该消息会自动添加到当前Agent的消息列表，并放入历史队列。

        :return: 助手消息
        """
        assistant_msg: AssistantMessage = self.client.send(self.agent.key_chain, self.agent.messages, self.agent.info.tools, self.agent.info.with_stream)
        self.history_queue.put((self.agent.key_chain, assistant_msg))
        self.agent.messages.append(assistant_msg)
        return assistant_msg

    def sub_agent(self, agent: Agent):
        """
        切换到子Agent：深拷贝当前Agent入栈，并将当前Agent设为子Agent。
        如果 memory 中存在该子 Agent 的 key_chain 对应的历史消息，则直接替换子 Agent 的消息列表（不拼接），以实现有状态记忆。

        :param agent: 子Agent实例
        """
        previous_agent: Agent = copy.deepcopy(self.agent)
        self.stack.append(previous_agent)
        self.agent = agent
        history: Optional[List[Message]] = self.memory.get(self.agent.key_chain)
        if history:
            self.agent.messages = copy.deepcopy(history)

    def back(self):
        """
        返回到父Agent：将当前 Agent 的消息历史保存到 memory 中，然后弹出栈顶，恢复父Agent为当前Agent。

        若当前 Agent 的 `rm_rf_memory` 为 True，则不保存记忆，而是调用 `rm_rf()` 方法
        递归删除以当前 Agent 的 `KeyChain` 为前缀的所有 memory 条目。
        """
        if not self.agent.info.rm_rf_memory:
            self.memory[self.agent.key_chain] = copy.deepcopy(self.agent.messages)
        else:
            self.rm_rf()
        self.agent = self.stack.pop()

    def rm_rf(self):
        """
        递归清理记忆缓存。

        删除 `self.memory` 字典中，键以当前 Agent 的 `KeyChain` 为前缀的所有条目。
        清理范围包括当前 Agent 自身，以及其所有后代子 Agent 的记忆。
        """
        key_chain_list: List[KeyChain] = list(self.memory.keys())
        current_key_chain: KeyChain = self.agent.key_chain
        length: int = len(current_key_chain)

        for key_chain in key_chain_list:
            if key_chain[:length] == current_key_chain and len(key_chain) > length:
                del self.memory[key_chain]
