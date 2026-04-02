import copy
from queue import Queue
from typing import Dict, Callable, Tuple, Optional, List

from oak_deepseek.agent import AgentFactory, Agent
from oak_deepseek.core import AgentCore
from oak_deepseek.tools import parse_tool_calls, ToolCall, parse_tool_call
from oak_deepseek.models import AssistantMessage, UserMessage, SystemMessage, ToolMessage


def init(core: AgentCore, task: str):
    """
    用于Agent初始化
    :param core: AgentCore，会话核心
    :param task: str，Agent要执行的任务
    :return:
    """
    if len(core.agent.messages) == 0:
        # 当前agent信息在引擎初始化，或调用子agent时在引擎中更新
        core.update(SystemMessage(content=core.agent.info.prompt))
        core.update(UserMessage(content=task))

def finish(core: AgentCore, call_info: ToolCall):
    """
    处理Agent的任务总结

    :param core: AgentCore，会话核心
    :param call_info: namedtuple("ToolCall", ["id", "name", "args"])，调用信息
    :return: 没有返回值，不过对应的分支可能要返回None
    """
    # 先生成子agent需要的工具信息
    tool_call_id: str = call_info.id
    args: Dict = call_info.args

    # 当前Agent收尾
    conclusion: str = args["conclusion"]
    core.update(ToolMessage(content=f"任务摘要：{conclusion}", tool_call_id=tool_call_id))

    # 返回父Agent，构造ToolMessage
    core.back()
    last_tool_call: Dict = core.agent.messages[-1].tool_calls[0]

    # tool_call_id从最后一条消息中取
    parent_call_id: str = parse_tool_call(last_tool_call)[0]
    core.update(ToolMessage(content=f"{conclusion}", tool_call_id=parent_call_id))

def new_agent(agent_factory: AgentFactory, core: AgentCore, call_info: ToolCall) -> str:
    """
    处理Agent调用子Agent

    :param agent_factory: 当前会话的AgentFactory，用于生成新的Agent实例
    :param core: AgentCore，会话核心
    :param call_info: namedtuple("ToolCall", ["id", "name", "args"])，调用信息
    :return: str，新Agent的任务，需要在循环体内返回
    """
    args: Dict = call_info[2]

    key: Tuple[str, str] = (args["agent"][0], args["agent"][1])
    task: str = args["task"]

    key_chain: List[Tuple[str, str]] = core.agent.key_chain
    key_chain.append(key)

    # 不需要深拷贝，build方法内是深拷贝
    agent: Agent = agent_factory.build(key_chain)
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
         task: str, tools: Dict[str, Callable], queue: Queue[str]) -> Optional[str]:
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
    while True:
        assistant_msg: AssistantMessage = core.send()
        if assistant_msg.tool_calls is not None:
            tool_queue: Queue[Tuple[str, str, Dict]] = parse_tool_calls(assistant_msg.tool_calls)
            while tool_queue.qsize() > 0:
                call_info: tuple[str, str, dict] = tool_queue.get(block=True)
                match call_info[1]:
                    case "wait_for_input":
                        content: str = queue.get(block=True)
                        core.update(ToolMessage(content=content, tool_call_id=call_info[0]))
                    case "finished":
                        finish(core, call_info)
                        return None
                    case "choose_agent":
                        # 这里返回的是子Agent的任务
                        return new_agent(agent_factory, core, call_info)
                    case _:
                        exec_tool(core, tools, call_info)