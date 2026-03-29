from queue import Queue
from typing import Dict, Callable, Tuple, Optional

from oak_deepseek.agent import AgentFactory
from oak_deepseek.core import AgentCore
from oak_deepseek.loops.branch import init, parse_tool_calls, exec_tool, new_agent, finish
from oak_deepseek.models import AssistantMessage, UserMessage


def re_act(engine: AgentCore, agent_factory: AgentFactory, task: str, tools: Dict[str, Callable]) -> Optional[str]:
    init(engine, task)
    while True:
        assistant_msg: AssistantMessage = engine.send()
        if assistant_msg.tool_calls is not None:
            tool_queue: Queue[Tuple[str, str, Dict]] = parse_tool_calls(assistant_msg.tool_calls)
            while tool_queue.qsize() > 0:
                # name
                call_info: tuple[str, str, dict] = tool_queue.get(block=True)
                match call_info[1]:
                    case "finished":
                        finish(engine, call_info)
                        return None
                    case "choose_agent":
                        return new_agent(agent_factory, engine, call_info)
                    case _:
                        exec_tool(engine, tools, call_info)
        else:
            pass


def reactive_rea_ct(engine: AgentCore, agent_factory: AgentFactory, task: str, tools: Dict[str, Callable], queue: Queue[str]) -> Optional[str]:
    init(engine, task)
    while True:
        assistant_msg: AssistantMessage = engine.send()
        if assistant_msg.tool_calls is not None:
            tool_queue: Queue[Tuple[str, str, Dict]] = parse_tool_calls(assistant_msg.tool_calls)
            while tool_queue.qsize() > 0:
                # name
                call_info: tuple[str, str, dict] = tool_queue.get(block=True)
                match call_info[1]:
                    case "finished":
                        finish(engine, call_info)
                    case "choose_agent":
                        return new_agent(agent_factory, engine, call_info)
                    case _:
                        exec_tool(engine, tools, call_info)
        else:
            engine.update(UserMessage(content=queue.get(block=True)))