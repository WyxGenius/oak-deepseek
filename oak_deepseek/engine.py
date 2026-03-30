from queue import Queue
from typing import List, Callable, Literal, Optional, Tuple, Dict, Union
import warnings

from oak_deepseek.agent import AgentInfo, AgentFactory, Agent
from oak_deepseek.client import RequestResponsePair
from oak_deepseek.core import AgentCore
from oak_deepseek.loops import re_act, reactive_rea_ct
from oak_deepseek.models import Tool, Message, AssistantMessage, ToolMessage, UserMessage, SystemMessage
from oak_deepseek.tools import standardize_tool, parse_tool_calls, ToolCall, parse_tool_call, if_finished_in_message


class AgentEngine:
    """
    核心类，用于创建和运行Agent系统。
    引擎本身是无状态的，只保存注册信息。实际运行状态由AgentCore管理，
    支持断点重连：通过create_core传入历史消息列表即可恢复之前的执行状态。
    """
    def __init__(self):
        self.agent_factory: AgentFactory = AgentFactory()
        self.tools: Dict[str, Callable] = {}

    # 快速注册
    def create_agent(self,
                     key: Tuple[str, str],
                     description: str,
                     prompt: str,
                     tools: Optional[List[Callable]]=None,
                     loop: Literal["ReactiveReAct", "ReAct"]="ReAct",
                     sub_agents: Optional[List[Tuple[str, str]]]=None):
        """
        注册一个Agent至引擎，以后可以用唯一命名空间+名字来指定。

        :param key: 命名空间+名字，例如 ("sys", "math_agent")
        :param description: Agent的简短描述
        :param prompt: Agent的系统提示词
        :param tools: 可选，Agent可调用的工具函数列表
        :param loop: 消息循环模式，可选 "ReAct" 或 "ReactiveReAct"
        :param sub_agents: 可选，可调用的子Agent列表，每个元素为子Agent的key
        :return: None
        """

        tools_info: Optional[List[Tool]] = []

        if tools is not None:
            for tool in tools:
                self.tools[tool.__name__] = tool
                tools_info.append(standardize_tool(tool))

        self.agent_factory.register_agent(key, AgentInfo(
            description=description, prompt=prompt, tools=tools_info, loop=loop, sub_agents=sub_agents
        ))

    def create_core(self, key: Union[Tuple[str, str], List[Tuple[Tuple[str, str], Message]]],
                    history_queue: Queue,
                    raw_response_queue: Optional[Queue[RequestResponsePair]] = None,
                    api_key: Optional[str] = None
                    ) -> AgentCore:
        """
        初始化引擎，指定入口Agent和消息记录队列。

        支持两种模式：

        - 正常启动：传入一个元组 (namespace, name) 作为入口Agent的key。
        - 断点恢复：传入一个历史消息列表，每个元素为 (agent_key, message)，按时间顺序排列。
          引擎会根据消息中的agent_key重建调用栈和各Agent的消息历史。

        :param key: 启动方式。

        - 若为元组 (namespace, name)，表示正常启动，该元组为入口 Agent 的 key。
        - 若为列表，表示断点恢复模式。列表元素为 (agent_key, message)，按时间顺序排列。其中 agent_key 是产生该消息的 Agent 的标识 (namespace, name)，message 是具体的 Message 对象。

        :param history_queue: 消息输出队列，运行期间产生的所有消息（附带所属Agent key）都会被放入此队列
        :param raw_response_queue: 可选，用于输出原始请求/响应对的队列
        :param api_key: DeepSeek API密钥，默认从环境变量DEEPSEEK_API_KEY读取
        :return: 已初始化的AgentCore实例
        """
        if isinstance(key, tuple):
            return AgentCore(self.agent_factory.build(key), history_queue, api_key=api_key, raw_response_queue=raw_response_queue)
        else:
            owner: List[Tuple[str, str]] = [key[0][0]]
            core: AgentCore = AgentCore(self.agent_factory.build(owner[-1]),
                                        history_queue, api_key=api_key, raw_response_queue=raw_response_queue)

            # 获取最后一条消息和所有者
            last_message: Message = key[-1][1]
            last_key: Tuple[str, str] = key[-1][0]

            # 最后一条是UserMessage：不用管
            if isinstance(last_message, UserMessage):
                pass

            # 最后一条是SystemMessage：说明任务信息丢失，可以直接结束
            elif isinstance(last_message, SystemMessage):
                recovery_msg: UserMessage = UserMessage(
                    content="执行刚刚被打断，现已重启。请直接调用finished工具，让上级重新委派任务，并简洁地说明原因"
                )
                key.append((last_key, recovery_msg))
                core.history_queue.put((last_key, recovery_msg))

            # 最后一条是AssistantMessage：先判断工具调用情况
            elif isinstance(last_message, AssistantMessage):
                if last_message.tool_calls is None:
                    pass
                else:
                    if if_finished_in_message(last_message):
                        recovery_msg: ToolMessage = ToolMessage(
                            content="执行刚刚被打断，现已重启。请重新调用finished",
                            tool_call_id=parse_tool_call(last_message.tool_calls[0]).id
                        )
                        key.append((last_key, recovery_msg))
                        core.history_queue.put((last_key, recovery_msg))
                    else:
                        tools_queue: Queue[ToolCall] = parse_tool_calls(last_message.tool_calls)
                        while tools_queue.qsize() > 0:
                            recovery_msg: ToolMessage = ToolMessage(
                                content="执行刚刚被打断，现已重启。请根据工具幂等性决定是否重试",
                                tool_call_id=tools_queue.get().id
                            )
                            key.append((last_key, recovery_msg))
                            core.history_queue.put((last_key, recovery_msg))

            # 最后一条是ToolMessage：倒序遍历至AssistantMessage后再决定
            elif isinstance(last_message, ToolMessage):
                # 倒走遍历至AssistanMessage
                last_messages: List[Tuple[Tuple[str, str], Message]] = []
                for snapshot in reversed(key):
                    if isinstance(snapshot[1], AssistantMessage):
                        last_messages.append(snapshot)
                        break
                    last_messages.append(snapshot)

                # 检查倒数第二条消息是不是带finished的AssistantMessage
                msg_2nd: Message = last_messages[1][1]
                key_2nd: Tuple[str, str] = last_messages[1][0]
                j = if_finished_in_message(msg_2nd)

                if isinstance(msg_2nd, AssistantMessage) and j:
                    # 让agent重新调用finished
                    recovery_msg: UserMessage = UserMessage(
                            content="执行刚刚被打断，现已重启。你需要重新调用finished生成总结",
                        )
                    key.append((key_2nd, recovery_msg))
                    core.history_queue.put((key_2nd, recovery_msg))
                else:
                    last_assistant_message: Tuple[Tuple[str, str], AssistantMessage] = last_messages.pop()
                    last_tool_messages: List[Tuple[Tuple[str, str], ToolMessage]] = [last_tool_message for
                                                                                     last_tool_message in
                                                                                     reversed(last_messages)]

                    completed_count: int = len(last_tool_messages)
                    tools_queue: Queue[ToolCall] = parse_tool_calls(last_assistant_message[1].tool_calls)

                    # 调用的工具比完成的多，就触发补全
                    if tools_queue.qsize() > completed_count:
                        while completed_count > 0:
                            tools_queue.get_nowait()
                            completed_count -= 1

                        while tools_queue.qsize() > 0:
                            recovery_msg: ToolMessage = ToolMessage(
                                content="执行刚刚被打断，现已重启。请根据工具幂等性决定是否重试。",
                                tool_call_id=tools_queue.get().id
                            )
                            key.append((last_key, recovery_msg))
                            core.history_queue.put((last_key, recovery_msg))

            # 消息已补全，正式开始恢复
            for snapshot in key:
                if snapshot[0] == owner[-1]:
                    # 直接写入消息
                    core.agent.messages.append(snapshot[1])
                else:
                    if len(owner) > 1 and snapshot[0] == owner[-2]:
                        # 返回父agent写入消息
                        core.back()
                        core.agent.messages.append(snapshot[1])
                        # 更新调用链
                        owner.pop()
                    else:
                        # 创建子agent写入消息
                        sub_agent: Agent = self.agent_factory.build(snapshot[0])
                        core.sub_agent(sub_agent)
                        core.agent.messages.append(snapshot[1])
                        # 更新调用链
                        owner.append(snapshot[0])

            return core

    def run(self, input_queue: Queue[str],
            key: Union[Tuple[str, str], List[Tuple[Tuple[str, str], Message]]],
            history_queue: Queue,
            raw_response_queue: Optional[Queue[RequestResponsePair]] = None,
            api_key: Optional[str] = None
            ):
        """
        开始任务，阻塞直到所有Agent完成工作。
        此方法会：

        1. 调用 create_core 创建 AgentCore 实例（支持正常启动或历史恢复）。
        2. 正常启动时，从 input_queue 获取初始任务；恢复时，直接从历史中恢复执行状态，不再读取初始任务。恢复时，传入循环函数的 task 参数为 None，会被 init 函数自动忽略。
        3. 根据当前Agent的循环模式，反复调用相应的循环函数，直到调用栈为空。

        注意：在恢复模式下，input_queue 仍然用于在 ReactiveReAct 循环中接收新的用户输入。

        :param input_queue: 用户输入队列，用于接收新消息（在ReactiveReAct模式下需要）
        :param key: 同create_core的key参数，用于确定启动方式
        :param history_queue: 消息输出队列，同create_core
        :param raw_response_queue: 可选，原始请求/响应队列
        :param api_key: DeepSeek API密钥
        :return: None
        """
        core: AgentCore = self.create_core(key=key, history_queue=history_queue, raw_response_queue=raw_response_queue, api_key=api_key)
        agent_count: int = len(core.stack) + 1

        task: Optional[str]
        if core.agent.messages is not None:
            task = None
        else:
            task = input_queue.get(block=True)
        while agent_count > 0:
            return_value = None
            # 检查当前Agent的工作模式
            if core.agent.info.loop == "ReAct":
                return_value: Optional[str] = re_act(core, self.agent_factory, task, self.tools)
            if core.agent.info.loop == "ReactiveReAct":
                return_value: Optional[str] = reactive_rea_ct(core, self.agent_factory, task, self.tools, input_queue)

            # 有返回值说明调用了子Agent，返回值是子Agent的任务
            if return_value is not None:
                task = return_value
                agent_count += 1
            else:
                agent_count += -1