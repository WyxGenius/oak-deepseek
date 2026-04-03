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
((('sys', 'sys'),), SystemMessage(role='system', content='你是一个严谨细致的数学计算助手，任何计算仅使用工具完成\n\n**你可以让这些AI Agent辅助你完成任务：**\n\n(namespace=sys, name=sub)\n简介：能进行乘法计算')),
((('sys', 'sys'),), UserMessage(role='user', content='计算(679+678)*(2+78)')),
((('sys', 'sys'),), AssistantMessage(role='assistant', content='', tool_calls=[{'index': 0, 'id': 'call_00_xkvsSUYQRsZn9DTCz4dH4RDL', 'type': 'function', 'function': {'name': 'add', 'arguments': '{"a": 679, "b": 678}'}}], reasoning_content='用户要求计算 (679+678)*(2+78)。我需要逐步计算。首先计算括号内的加法：679+678 和 2+78。我可以使用加法工具来计算这两个和。\n\n让我先计算 679+678。')),
((('sys', 'sys'),), ToolMessage(role='tool', content='1357', tool_call_id='call_00_xkvsSUYQRsZn9DTCz4dH4RDL')),
((('sys', 'sys'),), AssistantMessage(role='assistant', content='', tool_calls=[{'index': 0, 'id': 'call_00_Z6r4fy7PgZ3HvCWVIH73MP7g', 'type': 'function', 'function': {'name': 'add', 'arguments': '{"a": 2, "b": 78}'}}], reasoning_content='第一个和是1357。现在计算第二个和：2+78。')),
((('sys', 'sys'),), ToolMessage(role='tool', content='80', tool_call_id='call_00_Z6r4fy7PgZ3HvCWVIH73MP7g')),
#((('sys', 'sys'),), AssistantMessage(role='assistant', content='', tool_calls=[{'index': 0, 'id': 'call_00_RXE4X2ettuo7Y4rxvK8i7AUL', 'type': 'function', 'function': {'name': 'choose_agent', 'arguments': '{"agent": ["sys", "sub"], "task": "计算1357乘以80的结果"}'}}], reasoning_content='第二个和是80。现在需要计算1357乘以80。但我没有直接的乘法工具。不过，我可以使用sub这个AI Agent来进行乘法计算。根据系统提示，有一个namespace=sys, name=sub的AI Agent可以执行乘法计算。我需要使用choose_agent工具来调用它。让我调用这个AI Agent来计算1357 * 80。')),
#((('sys', 'sys'), ('sys', 'sub')), SystemMessage(role='system', content='你是一个严谨细致的数学计算助手，任何计算仅使用工具完成')),
#((('sys', 'sys'), ('sys', 'sub')), UserMessage(role='user', content='计算1357乘以80的结果')),
#((('sys', 'sys'), ('sys', 'sub')), AssistantMessage(role='assistant', content='', tool_calls=[{'index': 0, 'id': 'call_00_6C23Eruxiz71Qd3KP2fF80ai', 'type': 'function', 'function': {'name': 'mul', 'arguments': '{"a": 1357, "b": 80}'}}], reasoning_content='用户要求计算1357乘以80。这是一个简单的乘法运算，我可以使用乘法工具来完成。我需要调用乘法函数，参数a=1357，b=80。')),
#((('sys', 'sys'), ('sys', 'sub')), ToolMessage(role='tool', content='108560', tool_call_id='call_00_6C23Eruxiz71Qd3KP2fF80ai')),
#((('sys', 'sys'), ('sys', 'sub')), AssistantMessage(role='assistant', content='1357乘以80的结果是**108560**。', tool_calls=None, reasoning_content='工具返回的结果是108560。这个结果看起来是正确的：1357乘以80等于108560。现在我可以将这个结果返回给用户。不需要额外的解释，因为用户只要求计算结果。')),
#((('sys', 'sys'),), ToolMessage(role='tool', content='1357乘以80的结果是**108560**。', tool_call_id='call_00_RXE4X2ettuo7Y4rxvK8i7AUL')),
#((('sys', 'sys'),), AssistantMessage(role='assistant', content='根据计算步骤：\n\n1. 计算第一个括号：\\(679 + 678 = 1357\\)\n2. 计算第二个括号：\\(2 + 78 = 80\\)\n3. 相乘：\\(1357 \\times 80 = 108560\\)\n\n因此，\\((679+678) \\times (2+78) = 108560\\)。', tool_calls=None, reasoning_content='AI Agent返回了结果：108560。所以(679+678)*(2+78) = 108560。\n\n现在我需要给出最终答案。我应该展示完整的计算步骤。让我组织一下回答。\n\n首先计算679+678=1357，然后计算2+78=80，最后计算1357*80=108560。\n\n最终答案是108560。')),
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