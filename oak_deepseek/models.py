from typing import List, Optional, Literal, Union
from pydantic import BaseModel, Field, model_validator

from oak_deepseek.types import Message, Tool


class Thinking(BaseModel):
    type: Literal["enabled", "disabled"] = "enabled"
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

class ToolChoiceFunction(BaseModel):
    name: str = Field(..., description="要调用的函数名称。")

class ToolChoice(BaseModel):
    type: Literal["function"] = "function"  # 固定值
    function: ToolChoiceFunction

########################################################################################################################

class DeepSeekRequestBody(BaseModel):
    messages: List[Message]
    model: Literal["deepseek-reasoner"] = "deepseek-v4-flash"
    thinking: Thinking = Thinking.enable()
    max_tokens: Optional[int] = None
    response_format: Optional[ResponseFormat] = Field(default_factory=ResponseFormat.text)
    stop: Optional[Union[List[str], str]] = None
    stream: bool = False
    stream_options: Optional[StreamOptions] = None
    tools: Optional[List[Tool]] = None
    tool_choice: Optional[
        Union[
            Literal["none", "auto", "required"],
            ToolChoice
        ]
    ] = None

    ############################################################################################

    @model_validator(mode='after')
    def validate_thinking(self):
        if self.thinking == Thinking.disable():
            raise ValueError("仅支持推理模式")
        return self

    @model_validator(mode='after')
    def set_max_tokens_by_model(self):
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