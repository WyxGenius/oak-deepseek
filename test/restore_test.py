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
    prompt="你是一个数学计算助手，会使用工具确保自己计算的正确性。",
    tools=[add],
    sub_agents=[("sys","sub")]
)

engine.create_agent(
    key=("sys","sub"),
    description="能进行乘法计算",
    prompt="你是一个数学计算助手，会使用工具确保自己计算的正确性",
    tools=[mul]
)
history_queue = Queue()

task_queue = Queue()
task_queue.put("计算(679+678)*(2+78)")

history = [
(('sys', 'sys'), SystemMessage(role='system', content='你是一个数学计算助手，会使用工具确保自己计算的正确性。\n\n**你可以让这些AI Agent辅助你完成任务：**\n\n(namespace=sys, name=sub)\n简介：能进行乘法计算')),
(('sys', 'sys'), UserMessage(role='user', content='计算(679+678)*(2+78)')),
(('sys', 'sys'), AssistantMessage(role='assistant', content='我来帮你计算 (679+678)*(2+78)。首先我需要计算括号内的加法。\n\n让我先计算 679+678：', tool_calls=[{'index': 0, 'id': 'call_00_FSmWUlO8lMbthk1KkwbtnSMv', 'type': 'function', 'function': {'name': 'add', 'arguments': '{"a": 679, "b": 678}'}}], reasoning_content=None)),
(('sys', 'sys'), ToolMessage(role='tool', content='1357', tool_call_id='call_00_FSmWUlO8lMbthk1KkwbtnSMv')),
(('sys', 'sys'), AssistantMessage(role='assistant', content='现在计算 2+78：', tool_calls=[{'index': 0, 'id': 'call_00_gbjW4HraKWh3XBIM2vUdfOKW', 'type': 'function', 'function': {'name': 'add', 'arguments': '{"a": 2, "b": 78}'}}], reasoning_content=None)),
(('sys', 'sys'), ToolMessage(role='tool', content='80', tool_call_id='call_00_gbjW4HraKWh3XBIM2vUdfOKW')),
(('sys', 'sys'), AssistantMessage(role='assistant', content='现在我得到了两个结果：1357 和 80。接下来我需要计算 1357 × 80。不过我没有直接的乘法工具，但我可以使用 sub 工具来进行乘法计算。\n\n让我调用 sub 工具来计算乘法：', tool_calls=[{'index': 0, 'id': 'call_00_AXDtlOv2zb8Z9YCwvs6erkzR', 'type': 'function', 'function': {'name': 'choose_agent', 'arguments': '{"agent": ["sys", "sub"], "task": "计算 1357 × 80"}'}}], reasoning_content=None)),
(('sys', 'sub'), SystemMessage(role='system', content='你是一个数学计算助手，会使用工具确保自己计算的正确性')),
(('sys', 'sub'), UserMessage(role='user', content='计算 1357 × 80')),
(('sys', 'sub'), AssistantMessage(role='assistant', content='我来帮你计算 1357 × 80。', tool_calls=[{'index': 0, 'id': 'call_00_MO7YaC8pg5Fap9UlN7qoQpMh', 'type': 'function', 'function': {'name': 'mul', 'arguments': '{"a": 1357, "b": 80}'}}], reasoning_content=None)),
(('sys', 'sub'), ToolMessage(role='tool', content='108560', tool_call_id='call_00_MO7YaC8pg5Fap9UlN7qoQpMh')),
]

def go():
    engine.run(task_queue, key=history, history_queue=history_queue, api_key=os.getenv("DEEPSEEK_API_KEY"))

def messages():
    while True:
        msg = history_queue.get(block=True)
        if msg is None:
            return
        print(msg)


if __name__ == "__main__":
    agent_thread = Thread(target=go)
    agent_thread.start()

    print_thread = Thread(target=messages)
    print_thread.start()

    agent_thread.join()
    history_queue.put(None)
    print_thread.join()