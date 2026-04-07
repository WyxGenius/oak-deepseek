# Oak DeepSeek

围绕 DeepSeek API 开发，是一个能运行多Agent、能从断点恢复、能追溯完整执行流程的轻量 AI Agent 框架。

## 设计亮点

- **多级 Agent 嵌套**：支持任意深度调用子 Agent，自动管理调用栈，无需手动处理返回。
- **断点恢复**：传入历史消息即可精确恢复现场，适合长时任务、容灾、调试。
- **完整流程追溯**：每条消息都带调用链 `key_chain`，顺序如何，属于哪一个 Agent 一目了然。
- **统一调用模型**：子 Agent 委托与普通工具调用接口完全一致，调度逻辑极简。
- **零依赖轻量**：核心代码约 1000 行左右，无第三方 Agent 框架依赖，安装即用。
- **可观测**：内置消息队列与原始请求队列，支持实时监控、持久化。

## 安装  
  
```bash
pip install oak-deepseek
```

## 快速开始

下面通过一个**多 Agent 协作计算表达式**的完整示例，一步步带你了解 Oak DeepSeek 的核心用法。

### 第一步：导入必要模块

```python
from queue import Queue  
from threading import Thread  
import os  
from oak_deepseek.engine import AgentEngine
```

- `Queue`：用于线程间传递任务输入和输出消息。
- `Thread`：让 Agent 引擎和消息打印并发运行。
- `os`：读取环境变量中的 API Key。
- `AgentEngine`：框架核心，负责注册 Agent 和启动运行。

### 第二步：定义一个普通工具函数

```python
def store_lookup(value_name: str) -> str:  
    """返回数值存储中的值"""  
    store = {"a": 2, "b": 3, "c": 4, "d": 5}  
    return str(store.get(value_name, "unknown"))
```

这个函数模拟一个查询数值的工具：输入变量名 `a`、`b` 等，返回对应的数字（字符串形式）。  

### 第三步：创建引擎并注册三个 Agent

```python
engine = AgentEngine()
```

#### 3.1 注册数值存储子 Agent（带工具）

```python
engine.create_agent(  
    key=("sys", "store"),  
    description="数值存储库，可以查询 a,b,c,d 的值",  
    prompt="你是一个数值存储助手。用户会询问某个变量的值，请调用工具 store_lookup 获取并返回。",  
    tools=[store_lookup]  
)
```

- `key=("sys", "store")`：全局唯一标识，命名空间 `sys`，名字 `store`。
- `description`：简短描述，会被注入到调用者的提示词中。
- `prompt`：系统提示词，告诉这个 Agent 如何使用 `store_lookup` 工具。
- `tools=[store_lookup]`：将该工具函数注册给此 Agent。

#### 3.2 注册计算子 Agent（无工具，只会询问父 Agent）

```python
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
    sub_agents=[]  # 没有子 Agent，只依赖父 Agent
```

- 这个 Agent **没有工具**，它的提示词明确要求它通过 `choose_agent` 向父 Agent 询问未知数的值。
- `sub_agents=[]`：不注入 `choose_agent` 工具（实际上框架会自动注入，但空列表不会影响行为，这里显式声明便于理解）。

#### 3.3 注册根 Agent（协调者）

```python
engine.create_agent(  
    key=("sys", "root"),  
    description="协调助手，负责将计算任务委派给 calculator，并为 calculator 提供数值查询服务",  
    prompt=(  
        "你是一个协调助手。用户会要求你计算表达式。\n"  
        "你应该将计算任务委派给 calculator 子 Agent，并等待它的结果。\n"  
        "同时，当 calculator 向你询问某个数值时，你需要向 store 子 Agent 查询该数值，然后将结果返回给 calculator。\n"  
        "不要自己编造数值，必须通过 store 获取。"  
    ),  
    sub_agents=[("sys", "calculator"), ("sys", "store")]  
)
```

- `sub_agents` 列表告诉框架：根 Agent 可以调用 `calculator` 和 `store` 这两个子 Agent。
- 框架会自动为根 Agent 注入 `choose_agent` 工具，并将子 Agent 的描述信息拼接到提示词末尾。

### 第四步：准备消息队列并启动任务

```python
history_queue = Queue()   # 存放所有消息（含调用链），用于观察和断点恢复
task_queue = Queue()      # 存放用户输入的任务
task_queue.put("请计算表达式 (a * b) + (c * d)")
```

### 第五步：定义运行和打印函数

```python
def go():  
    engine.run(task_queue, key=("sys", "root"), history_queue=history_queue, api_key=os.getenv("DEEPSEEK_API_KEY"))  
  
def messages():  
    while True:  
        msg = history_queue.get(block=True)  
        if msg is None:  
            return  
        print(f"{msg},")
```

- `engine.run` 会阻塞，直到所有 Agent 完成工作（调用栈为空）。
- `history_queue` 中每一条消息都附带其所属 Agent 的完整 `key_chain`，例如 `((‘sys’,’root’), (‘sys’,’calculator’))` 表示这条消息是 calculator 发出的。

### 第六步：启动线程并运行

```python
if __name__ == "__main__":
    agent_thread = Thread(target=go)
    agent_thread.start()
    print_thread = Thread(target=messages)
    print_thread.start()
    agent_thread.join()
    history_queue.put(None)   # 通知打印线程退出
    print_thread.join()
```

### 预期效果

运行后，你可以观察到这三个 Agent 是如何协作完成表达式计算任务的。

### 下一步

- 创建一个新线程，通过 `task_queue` 给根Agent发送消息，体验多轮对话。
- 查看 `test/` 目录下的更多示例（数学计算、辩论赛、断点恢复等）。
- 修改提示词、增加新工具、自定义子 Agent，快速搭建自己的多智能体系统。

## 许可证  

MIT license