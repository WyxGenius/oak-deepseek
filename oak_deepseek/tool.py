import json
from collections import namedtuple
from typing import Callable, List, Dict, Literal, Any, Tuple
from pydantic import TypeAdapter

from oak_deepseek.models import Function, Tool


def standardize_tool(func: Callable) -> Tool:
    return Tool(
        type="function",
        function=Function(
            name=func.__name__,
            description=func.__doc__ or None,
            parameters=TypeAdapter(func).json_schema()
        )
    )

def standardize_tools(funcs: List[Callable]) -> List[Tool]:
    return [standardize_tool(func) for func in funcs]


# 命名元组，用起来更方便
ToolCall = namedtuple("ToolCall", ["id", "name", "args"])

def parse_tool_call(tool_call: Dict) -> ToolCall:
    """
    用来提取单个工具的核心数据
    :param tool_call: Dict，tool_calls字段中的单个元素
    :return: Tuple[str, str, Dict]，依次是id，函数名和参数
    """
    tool_call_id: str = tool_call["id"]
    call_data: Dict[Literal["name", "arguments"], Any] = tool_call["function"]

    name: str = call_data["name"]
    args: Dict = json.loads(call_data["arguments"])



    return ToolCall(id=tool_call_id, name=name, args=args)