import copy
from queue import Queue
from typing import Dict, Callable, Tuple, Optional, List

from oak_deepseek.agent import AgentFactory, Agent
from oak_deepseek.core import AgentCore
from oak_deepseek.tools import parse_tool_calls, ToolCall, parse_tool_call, is_finished
from oak_deepseek.models import AssistantMessage, UserMessage, SystemMessage, ToolMessage


def init(core: AgentCore, task: Optional[str]):
    """
    用于Agent初始化
    :param core: AgentCore，会话核心
    :param task: Optional[str]，Agent要执行的任务
    :return:
    """
    if len(core.agent.messages) == 0:
        # 当前agent信息在引擎初始化，或调用子agent时在引擎中更新
        core.update(SystemMessage(content=core.agent.info.prompt))
    if task is not None:
        core.update(UserMessage(content=task))

def new_agent(agent_factory: AgentFactory, core: AgentCore, call_info: ToolCall) -> str:
    """
    处理Agent调用子Agent

    :param agent_factory: 当前会话的AgentFactory，用于生成新的Agent实例
    :param core: AgentCore，会话核心
    :param call_info: namedtuple("ToolCall", ["id", "name", "args"])，调用信息
    :return: str，新Agent的任务，需要在循环体内返回
    :raises KeyError: 如果指定的子 Agent 未在 AgentFactory 中注册。
    """
    args: Dict = call_info[2]

    key: Tuple[str, str] = (args["agent"][0], args["agent"][1])
    task: str = args["task"]

    key_chain: Tuple[Tuple[str, str], ...] = core.agent.key_chain
    agent: Agent = agent_factory.build(key_chain + (key,))
    core.sub_agent(agent)
    return task

def exec_tool(core: AgentCore, tools: Dict[str, Callable], call_info: ToolCall):
    """
    直接执行工具，最简单的一集

    :param core: AgentCore，会话核心
    :param tools: Dict[str, Callable]，当前会话的可用工具
    :param call_info: namedtuple("ToolCall", ["id", "name", "args"])，调用信息
    :return: 没有返回值
    """
    tool_call_id: str = call_info[0]
    name: str = call_info[1]
    args: Dict = call_info[2]

    content: str = tools.get(name)(**args)
    core.update(ToolMessage(content=content, tool_call_id=tool_call_id))

########################################################################################################################

def main(core: AgentCore,
         agent_factory: AgentFactory,
         task: Optional[str], tools: Dict[str, Callable], queue: Queue[str]) -> Optional[str]:
    """
    消息循环。

    :param core: 当前运行的AgentCore
    :param agent_factory: Agent工厂
    :param task: 当前Agent的任务
    :param tools: 可用的工具字典
    :param queue: 用户输入队列
    :return: 如果调用了子Agent，返回子Agent的任务字符串；否则返回None
    """
    init(core, task)

    # 获取第一条助手消息
    assistant_msg: AssistantMessage

    if is_finished(core.agent.messages[-1]):
        assistant_msg = core.agent.messages[-1]
    else:
        assistant_msg = core.send()

    while True:
        # 处理助手消息
        if is_finished(assistant_msg):
            if len(core.agent.key_chain) > 1:
                core.back()
                last_tool_call: Dict = core.agent.messages[-1].tool_calls[0]
                parent_call_id: str = parse_tool_call(last_tool_call).id
                core.update(ToolMessage(content=f"{assistant_msg.content}", tool_call_id=parent_call_id))
                return None
            else:
                content: str = queue.get(block=True)
                core.update(UserMessage(content=content))

        elif assistant_msg.tool_calls is not None:
            tool_queue: Queue[ToolCall] = parse_tool_calls(assistant_msg.tool_calls)
            while tool_queue.qsize() > 0:
                call_info: ToolCall = tool_queue.get(block=True)
                match call_info.name:
                    case "choose_agent":
                        # 这里返回的是子Agent的任务
                        return new_agent(agent_factory, core, call_info)
                    case _:
                        exec_tool(core, tools, call_info)
        else:
            pass
        # 再次获取
        assistant_msg: AssistantMessage = core.send()