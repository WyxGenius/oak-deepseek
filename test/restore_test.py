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
    tools=[mul]
)
history_queue = Queue()

task_queue = Queue()
#task_queue.put("计算(679+678)*(2+78)")

history = [
([('sys', 'sys')], SystemMessage(role='system', content='你是一个严谨细致的数学计算助手，任何计算仅使用工具完成\n\n**你可以让这些AI Agent辅助你完成任务：**\n\n(namespace=sys, name=sub)\n简介：能进行乘法计算')),
#([('sys', 'sys')], UserMessage(role='user', content='计算(679+678)*(2+78)')),
#([('sys', 'sys')], AssistantMessage(role='assistant', content='我来帮您计算 (679+678)*(2+78)。首先我需要计算括号内的加法，然后进行乘法运算。\n\n让我先计算第一个括号内的加法：679 + 678。', tool_calls=[{'index': 0, 'id': 'call_00_RZTI6plraONJXimUoLFUekqF', 'type': 'function', 'function': {'name': 'add', 'arguments': '{"a": 679, "b": 678}'}}], reasoning_content=None)),
#([('sys', 'sys')], ToolMessage(role='tool', content='1357', tool_call_id='call_00_RZTI6plraONJXimUoLFUekqF')),
#([('sys', 'sys')], AssistantMessage(role='assistant', content='现在计算第二个括号内的加法：2 + 78。', tool_calls=[{'index': 0, 'id': 'call_00_1te9IDUVBWx3MCG7rD7QnKFx', 'type': 'function', 'function': {'name': 'add', 'arguments': '{"a": 2, "b": 78}'}}], reasoning_content=None)),
#([('sys', 'sys')], ToolMessage(role='tool', content='80', tool_call_id='call_00_1te9IDUVBWx3MCG7rD7QnKFx')),
#([('sys', 'sys')], AssistantMessage(role='assistant', content='现在我得到了两个中间结果：1357 和 80。接下来我需要计算这两个数的乘积。由于乘法计算需要使用专门的工具，我需要调用乘法计算助手来完成这个任务。', tool_calls=[{'index': 0, 'id': 'call_00_m5nRUvEeaLQMizySBPssyJ3d', 'type': 'function', 'function': {'name': 'choose_agent', 'arguments': '{"agent": ["sys", "sub"], "task": "请计算 1357 × 80 的结果"}'}}], reasoning_content=None)),
#([('sys', 'sys'), ('sys', 'sub')], SystemMessage(role='system', content='你是一个严谨细致的数学计算助手，任何计算仅使用工具完成')),
#([('sys', 'sys'), ('sys', 'sub')], UserMessage(role='user', content='请计算 1357 × 80 的结果')),
#([('sys', 'sys'), ('sys', 'sub')], AssistantMessage(role='assistant', content='我来帮你计算 1357 × 80 的结果。', tool_calls=[{'index': 0, 'id': 'call_00_k4WJRU0inPRQwGi6r8IGOcQ5', 'type': 'function', 'function': {'name': 'mul', 'arguments': '{"a": 1357, "b": 80}'}}], reasoning_content=None)),
#([('sys', 'sys'), ('sys', 'sub')], ToolMessage(role='tool', content='108560', tool_call_id='call_00_k4WJRU0inPRQwGi6r8IGOcQ5')),
#([('sys', 'sys'), ('sys', 'sub')], AssistantMessage(role='assistant', content='', tool_calls=[{'index': 0, 'id': 'call_00_rkuHGCTvZaf6FIJBHm4a0aOn', 'type': 'function', 'function': {'name': 'finished', 'arguments': '{"conclusion": "1357 × 80 = 108560"}'}}], reasoning_content=None)),
#([('sys', 'sys'), ('sys', 'sub')], ToolMessage(role='tool', content='任务摘要：1357 × 80 = 108560', tool_call_id='call_00_rkuHGCTvZaf6FIJBHm4a0aOn')),
#([('sys', 'sys')], ToolMessage(role='tool', content='1357 × 80 = 108560', tool_call_id='call_00_m5nRUvEeaLQMizySBPssyJ3d')),
#([('sys', 'sys')], AssistantMessage(role='assistant', content='很好！现在我已经完成了所有计算。让我总结一下：\n\n1. 679 + 678 = 1357\n2. 2 + 78 = 80  \n3. 1357 × 80 = 108,560', tool_calls=[{'index': 0, 'id': 'call_00_VXUQAyzvKCB3aoiJYSOZV4s9', 'type': 'function', 'function': {'name': 'wait_for_input', 'arguments': '{"msg": "计算 (679+678)*(2+78) 的结果如下：\\n\\n1. 首先计算括号内的加法：\\n   - 679 + 678 = 1,357\\n   - 2 + 78 = 80\\n\\n2. 然后计算乘积：\\n   - 1,357 × 80 = 108,560\\n\\n所以最终结果是：**108,560**"}'}}], reasoning_content=None)),
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