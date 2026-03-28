# Oak DeepSeek

一个轻量的 AI Agent 框架，基于 DeepSeek API 构建。通过队列通信和多线程设计，快速构建支持工具调用、子 Agent 协作的智能体。

## 安装

```bash
pip install oak-deepseek
```

## 快速开始
下面是一个完整示例，展示如何创建两个 Agent（主 Agent 和子 Agent）并让它们协作计算数学表达式。

```python
# 导入必要的模块
from queue import Queue          # 线程安全队列，用于传递消息和任务
from threading import Thread     # 多线程支持，让 Agent 运行与消息打印并行
import os                        # 读取环境变量（API 密钥）

from oak_deepseek.engine import AgentEngine  # 框架主类

# 定义工具函数（Agent 可调用的能力）
def add(a: int, b: int) -> str:
    """两个整数相加，返回字符串结果"""
    return f"{a + b}"

def mul(a: int, b: int) -> str:
    """两个整数相乘，返回字符串结果"""
    return f"{a * b}"

# 1. 创建引擎实例（整个会话的管理者）
engine = AgentEngine()

# 2. 注册主 Agent（负责整体任务调度）
engine.create_agent(
    key=("sys", "sys"),                          # Agent 唯一标识：(命名空间, 名字)
    description="一个能进行数学计算的agent",      # 简要描述，供其他 Agent 参考
    prompt="你是一个数学计算助手，会使用工具确保自己计算的正确性。",  # 系统提示词
    loop="reactive_enter",                       # 循环模式：reactive_enter 支持持续对话
    tools=[add],                                 # 可调用的工具列表
    sub_agents=[("sys", "sub")]                  # 可调用的子 Agent 列表
)

# 3. 注册子 Agent（专门负责乘法运算）
engine.create_agent(
    key=("sys", "sub"),
    description="能进行乘法计算",
    prompt="你是一个数学计算助手，会使用工具确保自己计算的正确性",
    tools=[mul]                                  # 子 Agent 只能调用乘法工具
)

# 4. 准备历史消息队列（框架会将所有消息按顺序放入此队列）
history_queue = Queue()

# 5. 初始化核心组件，指定入口 Agent 和 API 密钥
engine.init_core(
    key=("sys", "sys"),                          # 入口 Agent
    history_queue=history_queue,                 # 消息历史队列
    api_key=os.getenv("DEEPSEEK_API_KEY")        # 从环境变量读取 API 密钥
)

# 6. 准备任务输入队列（用户输入会放入此队列）
task_queue = Queue()
task_queue.put("计算(679+678)*(2+78)")            # 放入初始任务

# 7. 定义运行 Agent 的函数（将在独立线程中执行）
def go():
    engine.run(task_queue)                       # 阻塞运行，从 task_queue 获取用户输入

# 8. 定义打印历史消息的函数（在另一个线程中实时输出）
def messages():
    while True:
        msg = history_queue.get(block=True)      # 阻塞等待消息
        if msg is None:                          # 收到 None 表示会话结束
            return
        print(msg)                               # 打印每条消息（System/User/Assistant/Tool）

# 9. 启动两个线程
if __name__ == "__main__":
    agent_thread = Thread(target=go)             # Agent 运行线程
    agent_thread.start()

    print_thread = Thread(target=messages)       # 消息打印线程
    print_thread.start()

    agent_thread.join()                          # 等待 Agent 运行结束
    history_queue.put(None)                      # 向历史队列放入结束信号
    print_thread.join()                          # 等待打印线程结束
```

## 许可证

MIT license