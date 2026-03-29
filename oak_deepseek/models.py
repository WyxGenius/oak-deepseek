from typing import List, Dict, Optional, Literal, Any, Union, Annotated
from pydantic import BaseModel, Field, model_validator


########################################################################################################################

class SystemMessage(BaseModel):
    """系统消息。"""
    role: Literal["system"] = "system"
    content: str

class UserMessage(BaseModel):
    """用户消息。"""
    role: Literal["user"] = "user"
    content: str

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

########################################################################################################################

Message = Annotated[
    Union[SystemMessage, UserMessage, AssistantMessage, ToolMessage],
    Field(discriminator="role")
]

########################################################################################################################

class Thinking(BaseModel):
    type: Literal["enabled", "disabled"] = "disabled"
    @classmethod
    def enable(cls):
        return cls(type="enabled")

    @classmethod
    def disable(cls):
        return cls(type="disabled")

########################################################################################################################

class ResponseFormat(BaseModel):
    type: Literal["text", "json_object"] = "text"
    @classmethod
    def text(cls):
        return cls(type="text")
    @classmethod
    def json_object(cls):
        return cls(type="json_object")

########################################################################################################################

class StreamOptions(BaseModel):
    include_usage : bool
    @classmethod
    def true(cls):
        return cls(include_usage=True)
    @classmethod
    def false(cls):
        return cls(include_usage=False)

########################################################################################################################

class Function(BaseModel):
    """描述一个可供模型调用的函数。"""
    description: str = Field(...)
    name: str = Field(...)
    parameters: Optional[Dict[str, Any]] = Field(None, description="JSON Schema 对象")
    strict: bool = False

class Tool(BaseModel):
    type: Literal["function"] = "function"
    function: Function = Field(..., description="具体函数")

########################################################################################################################

class ToolChoiceFunction(BaseModel):
    name: str = Field(..., description="要调用的函数名称。")

class ToolChoice(BaseModel):
    type: Literal["function"] = "function"  # 固定值
    function: ToolChoiceFunction

########################################################################################################################

class DeepSeekRequestBody(BaseModel):
    messages: List[Message]
    model: Literal["deepseek-chat", "deepseek-reasoner"] = "deepseek-chat"
    thinking: Thinking = Thinking.disable()
    frequency_penalty: float = Field(0, ge=-2, le=2)
    max_tokens: Optional[int] = None
    presence_penalty: float = Field(0, ge=-2, le=2)
    response_format: Optional[ResponseFormat] = Field(default_factory=ResponseFormat.text)
    stop: Optional[Union[List[str], str]] = None
    stream: bool = False
    stream_options: Optional[StreamOptions] = None
    temperature: float = Field(1, ge=0, le=2)
    top_p: float = Field(1, ge=0, le=1)
    tools: Optional[List[Tool]] = None
    tool_choice: Optional[
        Union[
            Literal["none", "auto", "required"],
            ToolChoice
        ]
    ] = None
    logprobs : bool = False
    top_logprobs : Optional[int] = Field(None, ge=0, le=20)

    ############################################################################################

    @model_validator(mode='after')
    def set_max_tokens_by_model(self):
        if self.model == "deepseek-chat":
            max_allowed: int = 8192
            default_value: int = 4096
        else:
            max_allowed: int = 65536
            default_value: int = 32768

        if self.max_tokens is None:
            self.max_tokens = default_value
        elif self.max_tokens > max_allowed:
            raise ValueError(f"max_tokens for model '{self.model}' cannot exceed {max_allowed}")
        elif self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")

        return self

    @model_validator(mode='after')
    def check_stream_options(self):
        if not self.stream and self.stream_options:
            raise ValueError("设置了 stream_options 但是未开启 stream")
        return self

    @model_validator(mode='after')
    def check_logprobs(self):
        if not self.logprobs and self.top_logprobs is not None:
            raise ValueError("设置了 top_logprobs 但是未开启 logprobs")
        if self.logprobs and self.top_logprobs is None:
            raise ValueError("logprobs=True 时必须提供 top_logprobs")
        return self