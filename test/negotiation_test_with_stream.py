from queue import Queue
from threading import Thread
import os
from oak_deepseek.engine import AgentEngine
from oak_deepseek.models import AssistantMessage
from oak_deepseek.stream import Stream


# 数值存储子 Agent（无工具，仅通过 prompt 返回固定值）
def store_lookup(value_name: str) -> str:
    """返回数值存储中的值（模拟工具）"""
    store = {"a": 2, "b": 3, "c": 4, "d": 5}
    return str(store.get(value_name, "unknown"))

engine = AgentEngine()
# 注册数值存储子 Agent（带工具）
engine.create_agent(
    key=("sys", "store"),
    description="数值存储库，可以查询 a,b,c,d 的值",
    prompt="你是一个数值存储助手。用户会询问某个变量的值，请调用工具 store_lookup 获取并返回。",
    with_stream=True,
    tools=[store_lookup]
)

# 注册计算子 Agent（无工具，只能通过 choose_agent 向父 Agent 询问）
engine.create_agent(
    key=("sys", "calculator"),
    description="乘法计算专家，但不知道任何数值，必须向父 Agent 询问",
    prompt=(
        "你是一个乘法计算专家。当用户要求你计算表达式时，你**不知道任何数值**。\n"
        "你必须逐一询问每个未知数的值。\n"
        "例如：如果表达式包含 a, b, c, d，你需要依次询问 \"a 的值是多少？\"，得到回复后再问下一个。\n"
        "所有数值都获得后，再进行乘法计算并返回结果。\n"
        "注意：每次只问一个数值，不要一次性问多个。"
    ),
    with_stream=True,
    sub_agents=[]  # 没有子 Agent，只依赖父 Agent
)

# 注册根 Agent（负责协调，可以向 store 查询）
engine.create_agent(
    key=("sys", "root"),
    description="协调助手，负责将计算任务委派给 calculator，并为 calculator 提供数值查询服务",
    prompt=(
        "你是一个协调助手。用户会要求你计算表达式。\n"
        "你应该将计算任务委派给 calculator 子 Agent，并等待它的结果。\n"
        "同时，当 calculator 向你询问某个数值时，你需要向 store 子 Agent 查询该数值，然后将结果返回给 calculator。\n"
        "不要自己编造数值，必须通过 store 获取。"
    ),
    with_stream=True,
    sub_agents=[("sys", "calculator"), ("sys", "store")]
)

history_queue = Queue()

raw_response_queue = Queue()

task_queue = Queue()
task_queue.put("请计算表达式 (a * b) + (c * d)")

def go():
    engine.run(task_queue, key=("sys", "root"), history_queue=history_queue,
               api_key=os.getenv("DEEPSEEK_API_KEY"), raw_response_queue=raw_response_queue)

def messages():
    while True:
        msg = history_queue.get(block=True)
        if msg is None:
            return
        if isinstance(msg[1], AssistantMessage):
            continue
        #print(f"{msg},")

def print_stream():
    while True:
        raw_response = raw_response_queue.get(block=True)
        stream: Stream = raw_response[2]
        mode: str = ""
        for chunk in stream.get_from_chunks():
            if chunk[0] != mode:
                mode = chunk[0]
                print(f"\n\n********************** {chunk[0]} from {raw_response[0]} **********************\n")
            print(chunk[1], end='')

if __name__ == "__main__":
    agent_thread = Thread(target=go)
    agent_thread.start()

    print_thread = Thread(target=messages)
    print_thread.start()

    stream_thread = Thread(target=print_stream)
    stream_thread.start()

    agent_thread.join()
    history_queue.put(None)
    print_thread.join()
    stream_thread.join()