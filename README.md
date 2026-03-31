# Oak DeepSeek

一个轻量的 AI Agent 框架，基于 DeepSeek API 构建。通过队列通信和多线程设计，快速构建支持工具调用、子 Agent 协作的智能体。

## 安装

```bash
pip install oak-deepseek
```

## 快速开始
下面是一个完整示例，展示如何创建两个 Agent（主 Agent 和子 Agent）并让它们协作计算数学表达式。

```python
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
task_queue.put("计算(679+678)*(2+78)")

def go():
    engine.run(task_queue, key=("sys", "sys"), history_queue=history_queue, api_key=os.getenv("DEEPSEEK_API_KEY"))

def messages():
    while True:
        msg = history_queue.get(block=True)
        if msg is None:
            return
        print(f"{msg},")


if __name__ == "__main__":
    agent_thread = Thread(target=go)
    agent_thread.start()

    print_thread = Thread(target=messages)
    print_thread.start()

    agent_thread.join()
    history_queue.put(None)
    print_thread.join()
```

更多示例请参考test文件夹

## 许可证

MIT license