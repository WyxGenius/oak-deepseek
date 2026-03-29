from typing import Dict, Tuple, Callable

from oak_deepseek.agent import Agent, AgentFactory
from oak_deepseek.core import AgentCore
from oak_deepseek.models import SystemMessage, UserMessage, ToolMessage
from oak_deepseek.tools import parse_tool_call, ToolCall


def init(engine: AgentCore, task: str):
    """
    用于Agent初始化
    :param engine: AgentEngine，Agent所在的引擎
    :param task: str，Agent要执行的任务
    :return:
    """
    if len(engine.agent.messages) == 0:
        # 当前agent信息在引擎初始化，或调用子agent时在引擎中更新
        engine.update(SystemMessage(content=engine.agent.info.prompt))
        engine.update(UserMessage(content=task))


def finish(core: AgentCore, call_info: ToolCall):
    """
    处理Agent的任务总结
    :param core: AgentCore，当前会话的引擎
    :param call_info: namedtuple("ToolCall", ["id", "name", "args"])，调用信息
    :return: 没有返回值，不过对应的分支可能要返回None
    """
    tool_call_id: str = call_info.id
    args: Dict = call_info.args

    # 当前Agent收尾
    conclusion: str = args["conclusion"]
    core.update(ToolMessage(content=f"任务摘要：{conclusion}", tool_call_id=tool_call_id))

    # 判空
    if len(core.stack) > 0:
        # 准备父Agent工具content

        # 返回父Agent，构造ToolMessage
        core.back()
        last_tool_call: Dict = core.agent.messages[-1].tool_calls[0]

        # tool_call_id从最后一条消息中取
        tool_call_id: str = parse_tool_call(last_tool_call)[0]
        core.update(ToolMessage(content=f"AI Agent执行摘要：{conclusion}", tool_call_id=tool_call_id))

def new_agent(agent_factory: AgentFactory, engine: AgentCore, call_info: ToolCall) -> str:
    """
    处理Agent调用子Agent
    :param agent_factory: 当前会话的AgentFactory，用于生成新的Agent实例
    :param engine: AgentEngine，当前会话的引擎
    :param call_info: namedtuple("ToolCall", ["id", "name", "args"])，调用信息
    :return: str，新Agent的任务，需要在循环体内返回
    """
    args: Dict = call_info[2]

    key: Tuple[str, str] = (args["agent"][0], args["agent"][1])
    task: str = args["task"]
    agent: Agent = agent_factory.build(key)
    engine.sub_agent(agent)
    return task

def exec_tool(engine: AgentCore, tools: Dict[str, Callable], call_info: ToolCall):
    """
    直接执行工具，最简单的一集
    :param engine: AgentEngine，当前会话的引擎
    :param tools: Dict[str, Callable]，当前会话的可用工具
    :param call_info: namedtuple("ToolCall", ["id", "name", "args"])，调用信息
    :return: 没有返回值
    """
    tool_call_id: str = call_info[0]
    name: str = call_info[1]
    args: Dict = call_info[2]

    content: str = tools.get(name)(**args)
    engine.update(ToolMessage(content=content, tool_call_id=tool_call_id))