from queue import Queue
from threading import Thread
import os

from oak_deepseek.engine import AgentEngine

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

session: AgentEngine = AgentEngine()
session.create_agent(
    key=("sys","sys"),
    description="一个能进行数学计算的agent",
    prompt="你是一个数学计算助手，会使用工具确保自己计算的正确性。",
    loop="reactive_enter",
    tools=[add],
    sub_agents=[("sys","sub")]
)

session.create_agent(
    key=("sys","sub"),
    description="能进行乘法计算",
    prompt="你是一个数学计算助手，会使用工具确保自己计算的正确性",
    tools=[mul]
)
history_queue = Queue()
session.init_core(key=("sys", "sys"), history_queue=history_queue, api_key=os.getenv("DEEPSEEK_API_KEY"))

task_queue = Queue()
task_queue.put("计算(679+678)*(2+78)")

def go():
    session.run(task_queue)

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