from mailbox import Message
from queue import Queue
from threading import Thread
from typing import Tuple

from oak_deepseek.engine import AgentEngine
from oak_deepseek.models import AssistantMessage
from oak_deepseek.tools import parse_tool_call, if_wait_for_input_in_message

engine: AgentEngine = AgentEngine()
engine.create_agent(
    key=("system", "Alice"),
    description="AI聊天助手",
    prompt="你叫Alice，是一个极具思辨能力的辩手，你正在进行辩论赛，你持有AI的发展会使人类变聪明观点。",
)

engine.create_agent(
    key=("system", "Tim"),
    description="AI聊天助手",
    prompt="你叫Tim，是一个极具思辨能力的辩手，你正在进行辩论赛，持有AI的发展会使人类变笨观点。",
)

input_queue1: Queue[str] = Queue()
input_queue1.put("来进行一场辩论赛，你是正方。不要询问，直接开始")
input_queue2: Queue[str] = Queue()

history_queue1: Queue[Tuple[Tuple[str, str], Message]] = Queue()
history_queue2: Queue[Tuple[Tuple[str, str], Message]] = Queue()

def chat1():
    def run():
        engine.run(input_queue1, ("system", "Alice"), history_queue1)
    def send():
        while True:
            msg: Tuple[Tuple[str, str], Message] = history_queue1.get(block=True)
            print(msg)
            if isinstance(msg[1], AssistantMessage) and if_wait_for_input_in_message(msg[1]):
                call_info = parse_tool_call(msg[1].tool_calls[0])
                content: str = call_info.args["msg"]
                s = f"Alice说：{content}"
                #print(s)
                #print("\n\n" + "**********" * 10 + "\n\n")
                input_queue2.put(s)

    Thread(target=run).start()
    Thread(target=send).start()

def chat2():
    def run():
        engine.run(input_queue2, ("system", "Tim"), history_queue2)
    def send():
        while True:
            msg: Tuple[Tuple[str, str], Message] = history_queue2.get(block=True)
            print(msg)
            if isinstance(msg[1], AssistantMessage) and if_wait_for_input_in_message(msg[1]):
                call_info = parse_tool_call(msg[1].tool_calls[0])
                content: str = call_info.args["msg"]
                s = f"Tim说：{content}"
                #print(s)
                #print("\n\n" + "**********" * 10 + "\n\n")
                input_queue1.put(s)

    Thread(target=run).start()
    Thread(target=send).start()

if __name__ == "__main__":
    Thread(target=chat1).start()
    Thread(target=chat2).start()