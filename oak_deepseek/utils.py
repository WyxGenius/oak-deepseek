from queue import Queue
from threading import Thread
from typing import Union, Tuple, Optional

from oak_deepseek.client import ResponseData
from oak_deepseek.models import AssistantMessage, Message
from oak_deepseek.core import KeyChain


class StreamDisplay:
    def __init__(self, history_queue: Queue, raw_response_queue: Queue):
        self.history_queue: Queue[Optional[Tuple[KeyChain, Message]]] = history_queue
        self.raw_response_queue: Queue[Optional[ResponseData]] = raw_response_queue

        # 分别负责传输持久化信息和打印信息
        self.context_queue: Queue[Tuple[KeyChain, Message]] = Queue()
        self.display_queue: Queue[Union[ResponseData, Tuple[KeyChain, Message]]] = Queue()

        def get_history_message():
            while True:
                msg: Optional[Tuple[KeyChain, Message]] = self.history_queue.get(block=True)
                if msg is None:
                    break
                self.context_queue.put(msg)
                if not isinstance(msg[1], AssistantMessage):
                    self.display_queue.put(msg)

        def get_assistant_message():
            while True:
                s: Optional[ResponseData] = self.raw_response_queue.get(block=True)
                if s is None:
                    break
                self.display_queue.put(s)

        self.history_thread: Thread = Thread(target=get_history_message)
        self.history_thread.start()

        self.display_thread: Thread = Thread(target=get_assistant_message)
        self.display_thread.start()

    def get_context(self):
        return self.context_queue

    def get_display(self):
        return self.display_queue

    def quit(self):
        self.history_queue.put(None)
        self.raw_response_queue.put(None)


def is_stream(data: Union[ResponseData, Tuple[KeyChain, Message]]) -> bool:
    if isinstance(data, ResponseData):
        return True
    else:
        return False