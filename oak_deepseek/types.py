from collections import namedtuple
from typing import Literal, Optional, List, Dict, Annotated, Union, Any, Tuple

from pydantic import BaseModel, Field


class SystemMessage(BaseModel):
    """系统消息。"""
    role: Literal["system"] = "system"
    content: str
    name: Optional[str] = None


class UserMessage(BaseModel):
    """用户消息。"""
    role: Literal["user"] = "user"
    content: str
    name: Optional[str] = None


class AssistantMessage(BaseModel):
    """
    助手消息，可能包含工具调用和思考内容。

    :ivar content: 消息内容
    :ivar tool_calls: 可选，工具调用列表
    :ivar reasoning_content: 可选，思考模式下的推理内容
    """
    role: Literal["assistant"] = "assistant"
    content: Optional[str]
    tool_calls: Optional[List[Dict]] = None
    reasoning_content: Optional[str] = None


class ToolMessage(BaseModel):
    """工具执行结果消息。"""
    role: Literal["tool"] = "tool"
    content: str
    tool_call_id: str


Message = Annotated[
    Union[SystemMessage, UserMessage, AssistantMessage, ToolMessage],
    Field(discriminator="role")
]


class Function(BaseModel):
    """描述一个可供模型调用的函数。"""
    description: str = Field(...)
    name: str = Field(...)
    parameters: Optional[Dict[str, Any]] = Field(None, description="JSON Schema 对象")
    strict: bool = False


class Tool(BaseModel):
    type: Literal["function"] = "function"
    function: Function = Field(..., description="具体函数")


KeyChain = Tuple[Tuple[str, str], ...]

ResponseData = namedtuple("ResponseData", ["key_chain", "payload", "llm_response", "http_remnants"])
