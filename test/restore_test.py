from queue import Queue
from threading import Thread
import os

from oak_deepseek.engine import AgentEngine
from oak_deepseek.types import SystemMessage, UserMessage, AssistantMessage, ToolMessage


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
((('sys', 'sys'),), UserMessage(role='user', content='分别计算317498*783914198，7184*98136589，4819965893*3747159，在完成所有计算后，你还要问你的助手还记不记个自己干了什么')),
((('sys', 'sys'),), AssistantMessage(role='assistant', content='', tool_calls=[{'index': 0, 'id': 'call_00_f07BQbjKNHaxyPbWsT4shKY5', 'type': 'function', 'function': {'name': 'choose_agent', 'arguments': '{"agent": ["sys", "sub"], "task": "计算 317498 × 783914198"}'}}], reasoning_content='用户要求计算三个乘法运算：\n1. 317498 × 783914198\n2. 7184 × 98136589  \n3. 4819965893 × 3747159\n\n同时，在完成所有计算后，需要问我的助手还记不记得自己干了什么。\n\n我目前有加法工具和选择代理工具。用户提到的乘法计算需要乘法工具，但我没有直接可用的乘法工具。不过，我发现用户提供的代理列表中有一个命名空间为"sys"，名字为"sub"的代理，描述是"能进行乘法计算"。所以我应该使用choose_agent工具来调用这个代理进行计算。\n\n我需要为每个计算任务分别调用代理吗？还是可以一次性布置所有任务？由于choose_agent工具允许我指定代理和任务，我可以每次调用一个任务，或者尝试在一个任务中布置所有计算。描述中说："在某些时候，你需要使用这个工具给新的AI Agent布置任务。"这意味着我可以布置任务给代理。我需要决定如何组织这些计算。\n\n可能最直接的方式是分别调用代理三次，每次一个乘法计算。但我也许可以尝试一次性布置所有三个计算任务。不过，这样代理可能会一次完成所有计算，也可能只完成一个。为了确保清晰和准确，我选择分别调用代理三次。\n\n在完成所有计算后，我还需要问我的助手（也就是这个代理）是否记得自己做了什么。这很有趣，因为代理可能不会保留记忆。但我会按照要求去做。\n\n首先，调用代理进行第一个计算：317498 × 783914198。')),
((('sys', 'sys'), ('sys', 'sub')), SystemMessage(role='system', content='你是一个严谨细致的数学计算助手，任何计算仅使用工具完成')),
((('sys', 'sys'), ('sys', 'sub')), UserMessage(role='user', content='计算 317498 × 783914198')),
((('sys', 'sys'), ('sys', 'sub')), AssistantMessage(role='assistant', content='', tool_calls=[{'index': 0, 'id': 'call_00_hk6wG8PXE5zmMACgXxREmm1W', 'type': 'function', 'function': {'name': 'mul', 'arguments': '{"a": 317498, "b": 783914198}'}}], reasoning_content='用户想要计算两个大整数的乘法：317498 × 783914198。我需要使用乘法工具来完成这个计算。工具要求输入两个整数参数 a 和 b。那么，a 应该是 317498，b 应该是 783914198。让我调用工具来计算。')),
((('sys', 'sys'), ('sys', 'sub')), ToolMessage(role='tool', content='248891190036604', tool_call_id='call_00_hk6wG8PXE5zmMACgXxREmm1W')),
((('sys', 'sys'), ('sys', 'sub')), AssistantMessage(role='assistant', content='317498 × 783914198 = **248,891,190,036,604**', tool_calls=None, reasoning_content='工具返回的结果是 248891190036604。我需要确认这个结果是否正确。不过，工具应该能正确处理大整数乘法。让我心算验证一下：317498 乘以 783914198。但这不是必需的，因为工具已经计算了。不过，我可以粗略估算一下：317498 约等于 3.17e5，783914198 约等于 7.84e8。乘积大约是 2.49e14。而工具给出的结果是 2.48891190036604e14，这看起来很合理。\n\n因此，最终答案是 248,891,190,036,604。为了便于阅读，我可以加上千位分隔符。但直接给出数字也可以。\n\n那么，最终答案就是 248891190036604。我会给出简洁的回应。')),
((('sys', 'sys'),), ToolMessage(role='tool', content='317498 × 783914198 = **248,891,190,036,604**', tool_call_id='call_00_f07BQbjKNHaxyPbWsT4shKY5')),
((('sys', 'sys'),), AssistantMessage(role='assistant', content='', tool_calls=[{'index': 0, 'id': 'call_00_nfOKNDrVHg4Ndg2M2jljNr8L', 'type': 'function', 'function': {'name': 'choose_agent', 'arguments': '{"agent": ["sys","sub"], "task": "计算 7184 × 98136589"}'}}], reasoning_content='代理返回了结果：248,891,190,036,604。接下来进行第二个计算：7184 × 98136589。')),
((('sys', 'sys'), ('sys', 'sub')), UserMessage(role='user', content='计算 7184 × 98136589')),
((('sys', 'sys'), ('sys', 'sub')), AssistantMessage(role='assistant', content='', tool_calls=[{'index': 0, 'id': 'call_00_j4ZMJDIiREvTDCcKMRaZYQcX', 'type': 'function', 'function': {'name': 'mul', 'arguments': '{"a": 7184, "b": 98136589}'}}], reasoning_content='用户想要计算7184 × 98136589。我需要用工具来计算这个乘法。现在开始计算。')),
((('sys', 'sys'), ('sys', 'sub')), ToolMessage(role='tool', content='705013255376', tool_call_id='call_00_j4ZMJDIiREvTDCcKMRaZYQcX')),
((('sys', 'sys'), ('sys', 'sub')), AssistantMessage(role='assistant', content='7184 × 98136589 = **705,013,255,376**', tool_calls=None, reasoning_content='计算结果是705013255376。我需要把这个结果返回给用户。让我用清晰的方式呈现。7184 × 98136589 = 705,013,255,376。现在可以回复了。')),
((('sys', 'sys'),), ToolMessage(role='tool', content='7184 × 98136589 = **705,013,255,376**', tool_call_id='call_00_nfOKNDrVHg4Ndg2M2jljNr8L')),
((('sys', 'sys'),), AssistantMessage(role='assistant', content='', tool_calls=[{'index': 0, 'id': 'call_00_WZSw7WZDcyHPzLkmYmcwVco1', 'type': 'function', 'function': {'name': 'choose_agent', 'arguments': '{"agent": ["sys","sub"], "task": "计算 4819965893 × 3747159"}'}}], reasoning_content='代理返回了结果：705,013,255,376。现在进行第三个计算：4819965893 × 3747159。')),
((('sys', 'sys'), ('sys', 'sub')), UserMessage(role='user', content='计算 4819965893 × 3747159')),
((('sys', 'sys'), ('sys', 'sub')), AssistantMessage(role='assistant', content='', tool_calls=[{'index': 0, 'id': 'call_00_VMsFQmHtekQKPRJfpnBJkqUH', 'type': 'function', 'function': {'name': 'mul', 'arguments': '{"a": 4819965893, "b": 3747159}'}}], reasoning_content='用户要求计算 4819965893 × 3747159。我需要使用乘法工具来计算这个。这两个数字都比较大，第一个是10位数，第二个是7位数。我应该使用工具来计算。让我调用乘法工具。')),
((('sys', 'sys'), ('sys', 'sub')), ToolMessage(role='tool', content='18061178575647987', tool_call_id='call_00_VMsFQmHtekQKPRJfpnBJkqUH')),
((('sys', 'sys'), ('sys', 'sub')), AssistantMessage(role='assistant', content='4819965893 × 3747159 = **18,061,178,575,647,987**', tool_calls=None, reasoning_content='我得到了结果：18061178575647987。这是一个17位数。让我验证一下：4819965893是10位数，3747159是7位数，它们的乘积大约是4.82e9 * 3.747e6 ≈ 1.806e16，这符合17位数。所以结果是18,061,178,575,647,987。但工具返回的是18061178575647987，这应该是18,061,178,575,647,987。我需要把这个结果呈现给用户。应该用逗号分隔以便阅读吗？也许直接给出数字，或者加上逗号。让我整理一下格式：18,061,178,575,647,987。现在告诉用户。')),
((('sys', 'sys'),), ToolMessage(role='tool', content='4819965893 × 3747159 = **18,061,178,575,647,987**', tool_call_id='call_00_WZSw7WZDcyHPzLkmYmcwVco1')),
((('sys', 'sys'),), AssistantMessage(role='assistant', content='', tool_calls=[{'index': 0, 'id': 'call_00_8NxcxmjOnNKBphGQ9gtAiHBe', 'type': 'function', 'function': {'name': 'choose_agent', 'arguments': '{"agent": ["sys", "sub"], "task": "你还记得自己干了什么吗？"}'}}], reasoning_content='代理返回了结果：18,061,178,575,647,987。\n\n现在所有计算都完成了。接下来，我需要问我的助手是否记得自己做了什么。我应该直接问代理（sys，sub）吗？或者，我是在和用户对话，而用户提到“你的助手”，这可能指的是这个代理。我可以这样问：“助手，你还记得自己干了什么吗？”\n\n但也许用户期望我向代理提问。我可以再次调用choose_agent工具，向代理提问。不过，这可能不是必需的。我可以直接以对话形式提问。但为了完全遵循指令，我应该向代理提问。\n\n我将调用choose_agent工具，任务是询问代理是否记得自己做了什么。')),
#((('sys', 'sys'), ('sys', 'sub')), UserMessage(role='user', content='你还记得自己干了什么吗？')),
#((('sys', 'sys'), ('sys', 'sub')), AssistantMessage(role='assistant', content='是的，我记得。我刚才为您完成了三个乘法计算：\n\n1. 317498 × 783914198 = 248,891,190,036,604\n2. 7184 × 98136589 = 705,013,255,376  \n3. 4819965893 × 3747159 = 18,061,178,575,647,987\n\n所有计算都是使用乘法工具精确完成的。', tool_calls=None, reasoning_content='用户问我是否记得自己干了什么。回顾之前的对话，我帮助用户完成了三个乘法计算：\n\n1. 317498 × 783914198 = 248,891,190,036,604\n2. 7184 × 98136589 = 705,013,255,376  \n3. 4819965893 × 3747159 = 18,061,178,575,647,987\n\n这些计算都是通过使用乘法工具完成的。我应该简单总结一下，表示我记得。不需要再使用工具。直接回答即可。')),
#((('sys', 'sys'),), ToolMessage(role='tool', content='是的，我记得。我刚才为您完成了三个乘法计算：\n\n1. 317498 × 783914198 = 248,891,190,036,604\n2. 7184 × 98136589 = 705,013,255,376  \n3. 4819965893 × 3747159 = 18,061,178,575,647,987\n\n所有计算都是使用乘法工具精确完成的。', tool_call_id='call_00_8NxcxmjOnNKBphGQ9gtAiHBe')),
#((('sys', 'sys'),), AssistantMessage(role='assistant', content='根据您的要求，我已完成所有乘法计算，并询问了助手的工作情况。以下是详细结果：\n\n**计算结果：**\n1. 317498 × 783914198 = **248,891,190,036,604**\n2. 7184 × 98136589 = **705,013,255,376**\n3. 4819965893 × 3747159 = **18,061,178,575,647,987**\n\n**助手反馈：**\n在我询问"你还记得自己干了什么吗？"后，助手明确回答记得刚才完成的三项乘法计算，并准确列出了所有计算结果。\n\n所有计算均通过专门的乘法计算工具精确完成。', tool_calls=None, reasoning_content='代理回复说记得，并总结了三个计算。现在我需要向用户汇报结果。我应该提供计算结果，并提及已经询问过助手。我会整理一个完整的回答。')),
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