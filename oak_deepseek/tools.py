import json
from collections import namedtuple
from queue import Queue
from typing import Callable, List, Dict, Literal, Any
from pydantic import TypeAdapter

from oak_deepseek.models import Function, Tool, Message, AssistantMessage


def standardize_tool(func: Callable) -> Tool:
    """
    将 Python 函数转换为框架可用的 Tool 对象。

    :param func: 工具函数，需包含类型注解和文档字符串
    :return: Tool 对象
    """
    return Tool(
        type="function",
        function=Function(
            name=func.__name__,
            description=func.__doc__ or None,
            parameters=TypeAdapter(func).json_schema()
        )
    )

def standardize_tools(funcs: List[Callable]) -> List[Tool]:
    """
    批量转换工具函数。

    :param funcs: 工具函数列表
    :return: Tool 对象列表
    """
    return [standardize_tool(func) for func in funcs]


# 命名元组，用起来更方便
ToolCall = namedtuple("ToolCall", ["id", "name", "args"])

def parse_tool_call(tool_call: Dict) -> ToolCall:
    """
    用来提取单个工具的核心数据

    :param tool_call: Dict，tool_calls字段中的单个元素
    :return: ToolCall 实例，包含 id、name 和 args
    """
    tool_call_id: str = tool_call["id"]
    call_data: Dict[Literal["name", "arguments"], Any] = tool_call["function"]

    name: str = call_data["name"]
    args: Dict = json.loads(call_data["arguments"])

    return ToolCall(id=tool_call_id, name=name, args=args)

def parse_tool_calls(tool_calls: List[Dict]) -> Queue[ToolCall]:
    """
    将tool_calls字段的id，函数名和参数提取出来按顺序放队列里

    :param tool_calls: List[Dict]，原始tool_calls字段
    :return: Queue[namedtuple("ToolCall", ["id", "name", "args"])]，按顺序放好的调用信息
    """
    queue: Queue[ToolCall] = Queue()
    for tool_call in tool_calls:
        queue.put(parse_tool_call(tool_call))
    return queue

def is_finished(message: Message):
    """
    判断一条消息是否为最终答案（任务完成）。

    条件：
    - 是 AssistantMessage 类型
    - content 字段非空
    - 没有 tool_calls（即不再需要调用工具）

    :param message: 待判断的消息
    :return: 是否为最终答案
    """
    if not isinstance(message, AssistantMessage):
        return False
    if not message.content or message.tool_calls:
        return False
    return True


