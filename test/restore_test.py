from queue import Queue
from threading import Thread
import os

from oak_deepseek.engine import AgentEngine
from oak_deepseek.models import SystemMessage, UserMessage, AssistantMessage, ToolMessage

def add(a: int, b: int) -> str:
    """
    这个函数用于两个整数相加，并返回str结果给你
    :return: 加法结果
    """
    return f"{a + b}"

def mul(a: int, b: int) -> str:
    """
    这个函数用于两个整数相乘，并返回str结果给你
    :return: 乘法结果
    """
    return f"{a * b}"

engine: AgentEngine = AgentEngine()
engine.create_agent(
    key=("sys","sys"),
    description="一个能进行数学计算的agent",
    prompt="你是一个严谨细致的数学计算助手，任何计算仅使用工具完成",
    tools=[add],
    sub_agents=[("sys","sub")]
)

engine.create_agent(
    key=("sys","sub"),
    description="能进行乘法计算",
    prompt="你是一个严谨细致的数学计算助手，任何计算仅使用工具完成",
    loop="ReAct",
    tools=[mul]
)
history_queue = Queue()

task_queue = Queue()
#task_queue.put("计算(679+678)*(2+78)")

history = [
#(('sys', 'sys'), SystemMessage(role='system', content='你是一个严谨细致的数学计算助手，任何计算仅使用工具完成\n\n**你可以让这些AI Agent辅助你完成任务：**\n\n(namespace=sys, name=sub)\n简介：能进行乘法计算')),
#(('sys', 'sys'), UserMessage(role='user', content='计算(679+678)*(2+78)')),
#(('sys', 'sys'), AssistantMessage(role='assistant', content='我来帮您计算 (679+678)*(2+78)。首先我需要计算括号内的加法，然后再进行乘法运算。\n\n让我先计算第一个括号：679 + 678', tool_calls=[{'index': 0, 'id': 'call_00_FbCYOLxhOafImq2fbnVWjZQr', 'type': 'function', 'function': {'name': 'add', 'arguments': '{"a": 679, "b": 678}'}}], reasoning_content=None)),
#(('sys', 'sys'), ToolMessage(role='tool', content='1357', tool_call_id='call_00_FbCYOLxhOafImq2fbnVWjZQr')),
#(('sys', 'sys'), AssistantMessage(role='assistant', content='现在计算第二个括号：2 + 78', tool_calls=[{'index': 0, 'id': 'call_00_Cbt5D6WPeSyQOKZpbXxDoILQ', 'type': 'function', 'function': {'name': 'add', 'arguments': '{"a": 2, "b": 78}'}}], reasoning_content=None)),
#(('sys', 'sys'), ToolMessage(role='tool', content='80', tool_call_id='call_00_Cbt5D6WPeSyQOKZpbXxDoILQ')),
#(('sys', 'sys'), AssistantMessage(role='assistant', content='现在我需要计算 1357 × 80。我需要使用乘法工具来完成这个计算。', tool_calls=[{'index': 0, 'id': 'call_00_gRoqpRu190fkeoT5UQz8gVLe', 'type': 'function', 'function': {'name': 'choose_agent', 'arguments': '{"agent": ["sys", "sub"], "task": "请计算 1357 乘以 80 的结果"}'}}], reasoning_content=None)),
#(('sys', 'sub'), SystemMessage(role='system', content='你是一个严谨细致的数学计算助手，任何计算仅使用工具完成')),
#(('sys', 'sub'), UserMessage(role='user', content='请计算 1357 乘以 80 的结果')),
#(('sys', 'sub'), AssistantMessage(role='assistant', content='我来帮您计算 1357 乘以 80 的结果。', tool_calls=[{'index': 0, 'id': 'call_00_YsxGtjAAxLvNE52HwzprDQ2C', 'type': 'function', 'function': {'name': 'mul', 'arguments': '{"a": 1357, "b": 80}'}}], reasoning_content=None)),
#(('sys', 'sub'), ToolMessage(role='tool', content='108560', tool_call_id='call_00_YsxGtjAAxLvNE52HwzprDQ2C')),
#(('sys', 'sub'), AssistantMessage(role='assistant', content='', tool_calls=[{'index': 0, 'id': 'call_00_bpAEpZt4byR3JsYtzp9efBsZ', 'type': 'function', 'function': {'name': 'finished', 'arguments': '{"conclusion": "1357 乘以 80 的结果是 108,560。"}'}}], reasoning_content=None)),
#(('sys', 'sub'), ToolMessage(role='tool', content='任务摘要：1357 乘以 80 的结果是 108,560。', tool_call_id='call_00_bpAEpZt4byR3JsYtzp9efBsZ')),
#(('sys', 'sys'), ToolMessage(role='tool', content='1357 乘以 80 的结果是 108,560。', tool_call_id='call_00_gRoqpRu190fkeoT5UQz8gVLe')),
#(('sys', 'sys'), AssistantMessage(role='assistant', content='现在我来总结计算过程：', tool_calls=[{'index': 0, 'id': 'call_00_oy80nDkDBuJf4byB9UjbttVn', 'type': 'function', 'function': {'name': 'wait_for_input', 'arguments': '{"msg": "计算 (679+678)*(2+78) 的结果如下：\\n\\n**计算步骤：**\\n1. 计算第一个括号：679 + 678 = 1357\\n2. 计算第二个括号：2 + 78 = 80  \\n3. 计算乘法：1357 × 80 = 108,560\\n\\n**最终结果：**\\n(679+678)*(2+78) = 108,560"}'}}], reasoning_content=None))
]


def go():
    engine.run(task_queue, key=history, history_queue=history_queue, api_key=os.getenv("DEEPSEEK_API_KEY"))

def messages():
    while True:
        msg = history_queue.get(block=True)
        if msg is None:
            return
        print(msg)

def user():
    user_input: str = input()
    task_queue.put(user_input)


if __name__ == "__main__":
    agent_thread = Thread(target=go)
    agent_thread.start()

    print_thread = Thread(target=messages)
    print_thread.start()

    user_thread = Thread(target=user)
    user_thread.start()

    agent_thread.join()
    history_queue.put(None)
    print_thread.join()