from queue import Queue
from typing import Dict, Tuple, Callable

from oak_deepseek.agent import Agent, AgentFactory
from oak_deepseek.core import AgentCore
from oak_deepseek.models import SystemMessage, UserMessage, ToolMessage
from oak_deepseek.tool import parse_tool_call, ToolCall


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

def parse_tool_calls(tool_calls: ToolCall) -> Queue[ToolCall]:
    """
    将tool_calls字段的id，函数名和参数提取出来按顺序放队列里
    :param tool_calls: ToolCall，原始tool_calls字段
    :return: Queue[namedtuple("ToolCall", ["id", "name", "args"])]，按顺序放好的调用信息
    """
    queue: Queue[Tuple[str, str, Dict]] = Queue()
    for tool_call in tool_calls:
        queue.put(parse_tool_call(tool_call))
    return queue

def finish(engine: AgentCore, call_info: ToolCall):
    """
    处理Agent的任务总结
    :param engine: AgentEngine，当前会话的引擎
    :param call_info: namedtuple("ToolCall", ["id", "name", "args"])，调用信息
    :return: 没有返回值，不过对应的分支可能要返回None
    """
    tool_call_id: str = call_info[0]
    args: Dict = call_info[2]

    # 当前Agent收尾
    engine.update(ToolMessage(content="任务已完成", tool_call_id=tool_call_id))

    # 判空
    if len(engine.stack) > 0:
        # 准备父Agent工具content
        conclusion: str = args["conclusion"]

        # 返回父Agent，构造ToolMessage
        engine.back()
        last_tool_call: Dict = engine.agent.messages[-1].tool_calls[0]

        # tool_call_id从最后一条消息中取
        tool_call_id: str = parse_tool_call(last_tool_call)[0]
        engine.update(ToolMessage(content=conclusion, tool_call_id=tool_call_id))

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