from queue import Queue
from threading import Thread
from typing import Union, Tuple, Optional

from oak_deepseek.types import AssistantMessage, Message, KeyChain, ResponseData
from oak_deepseek.stream import Stream


class StreamDisplay:
    """
    流式信息分流与展示辅助器。

    该类作为 Oak DeepSeek 引擎与用户界面之间的适配层，负责将底层的原始消息队列
    （`history_queue` 和 `raw_response_queue`）重新路由为两个语义清晰的输出队列：

    - `context_queue`：存放完整的结构化消息，用于持久化、审计或断点恢复。
    - `display_queue`：存放需要实时展示的内容，包括非助手消息以及来自
      `raw_response_queue` 的原始响应数据（`ResponseData`）。

    `ResponseData` 是一个命名元组，定义于 `types.py`，包含以下字段：
        - key_chain : Tuple[Tuple[str, str], ...]
        - payload   : DeepSeekRequestBody (请求体)
        - llm_response : Union[Dict, Stream] (响应数据或流对象)
        - http_remnants : requests.Response (原始HTTP响应对象)

    使用 `is_response()` 和 `is_stream()` 可判断 `display_queue` 中元素的类型。
    ...
    """
    def __init__(self, history_queue: Queue, raw_response_queue: Queue):
        """
        初始化 StreamDisplay 实例并启动后台消费线程。

        :param history_queue: 来自 AgentEngine 的历史消息队列，元素为 Optional[Tuple[KeyChain, Message]]。
                              当收到 `None` 时，对应消费线程退出。
        :param raw_response_queue: 来自 AgentEngine 的原始响应队列，元素为 Optional[ResponseData]。
                                   当收到 `None` 时，对应消费线程退出。
        """
        self.history_queue: Queue[Optional[Tuple[KeyChain, Message]]] = history_queue
        self.raw_response_queue: Queue[Optional[ResponseData]] = raw_response_queue

        # 分别负责传输持久化信息和打印信息
        self.context_queue: Queue[Tuple[KeyChain, Message]] = Queue()
        self.display_queue: Queue[Union[ResponseData, Tuple[KeyChain, Message]]] = Queue()

        def get_history_message():
            """
            消费 history_queue 的后台线程函数。

            将每条历史消息同时放入 context_queue，若消息不是 AssistantMessage
            则额外放入 display_queue 以供展示。收到 `None` 时退出循环。
            """
            while True:
                msg: Optional[Tuple[KeyChain, Message]] = self.history_queue.get(block=True)
                if msg is None:
                    break
                self.context_queue.put(msg)
                if not isinstance(msg[1], AssistantMessage):
                    self.display_queue.put(msg)

        def get_assistant_message():
            """
            消费 raw_response_queue 的后台线程函数。

            将每个原始响应数据直接放入 display_queue。收到 `None` 时退出循环。
            """
            while True:
                s: Optional[ResponseData] = self.raw_response_queue.get(block=True)
                if s is None:
                    break
                self.display_queue.put(s)

        self.history_thread: Thread = Thread(target=get_history_message)
        self.history_thread.start()

        self.display_thread: Thread = Thread(target=get_assistant_message)
        self.display_thread.start()

    def get_context(self) -> Queue[Tuple[KeyChain, Message]]:
        """
        获取上下文队列。

        该队列中的每一条消息都是完整的 `(KeyChain, Message)` 元组，可用于：
            - 持久化到日志或数据库
            - 断点恢复时重建会话状态
            - 构建长期记忆或知识库

        :return: 存放持久化消息的队列。
        """
        return self.context_queue

    def get_display(self) -> Queue[Union[ResponseData, Tuple[KeyChain, Message]]]:
        """
        获取展示队列。

        该队列用于驱动用户界面（UI）的实时更新。队列元素分为两类：
            - `ResponseData`：来自原始响应的数据包，内部可能包含流式 `Stream` 对象。
            - `Tuple[KeyChain, Message]`：来自历史队列的非助手消息（如用户消息、工具消息）。

        使用者应结合 `is_response()` 和 `is_stream()` 函数判断具体类型，
        并自行决定渲染方式（如打字机效果、Markdown 解析等）。

        :return: 存放待展示数据的队列。
        """
        return self.display_queue

    def quit(self):
        """
        优雅关闭后台线程。

        向两个输入队列分别发送 `None` 停止信号。后台线程在消费到 `None` 后将退出循环。
        注意：该方法不会强制 `join` 线程，若需确保线程完全退出，可在调用后由主线程等待。
        """
        self.history_queue.put(None)
        self.raw_response_queue.put(None)


def is_response(data: Union[ResponseData, Tuple[KeyChain, Message]]) -> bool:
    """
    判断从 `display_queue` 取出的数据是否为 `ResponseData` 类型。

    该函数用于在消费 `display_queue` 时快速分流：
        - 若返回 `True`，说明数据是来自原始响应的数据包，可能包含流式内容。
        - 若返回 `False`，则数据是普通的结构化消息（`Tuple[KeyChain, Message]`）。

    :param data: 从 `display_queue` 中获取的数据项。
    :return: `True` 如果数据是 `ResponseData` 实例，否则 `False`。
    """
    return isinstance(data, ResponseData)

def is_stream(data: ResponseData) -> bool:
    """
    判断 `ResponseData` 内部是否包含流式 `Stream` 对象。

    当 `is_response()` 返回 `True` 时，可进一步调用本函数判断响应类型：
        - 若返回 `True`，表示响应是流式的，应通过 `Stream.get_from_chunks()` 迭代增量内容。
        - 若返回 `False`，表示响应是非流式的完整响应，可直接读取其内容。

    :param data: 已经确认为 `ResponseData` 类型的实例。
    :return: `True` 如果响应体是 `Stream` 对象，否则 `False`。
    """
    return isinstance(data.llm_response, Stream)