import copy
from queue import Queue
from typing import List, Callable, Literal, Optional, Tuple, Dict, Union

from oak_deepseek.agent import AgentInfo, AgentFactory, Agent
from oak_deepseek.client import RequestResponsePair
from oak_deepseek.core import AgentCore
from oak_deepseek.loop import main
from oak_deepseek.models import Tool, Message, AssistantMessage, ToolMessage, UserMessage, SystemMessage
from oak_deepseek.tools import standardize_tool, parse_tool_calls, ToolCall

recovery_prompt: str = """系统恢复：执行中断，最后一条 AssistantMessage 包含未完成的 tool_calls。

对于每个未完成的调用：

- 如果工具是幂等的，则**主动重试**：重新调用该工具。
- 否则（非幂等，如发送消息、扣款、修改状态），你无法确定是否已执行。请输出一条简短消息，向用户说明情况并请求指示，例如：
  “上一个操作可能已经执行，我不确定是否应该重试。请告知是否可以重新执行，或者跳过。”

之后等待用户输入，不要自动重试非幂等工具。"""

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
                     sub_agents: Optional[List[Tuple[str, str]]]=None):
        """
        注册一个Agent至引擎，以后可以用唯一命名空间+名字来指定。

        :param key: 命名空间+名字，例如 ("sys", "math_agent")
        :param description: Agent的简短描述
        :param prompt: Agent的系统提示词
        :param tools: 可选，Agent可调用的工具函数列表
        :param sub_agents: 可选，可调用的子Agent列表，每个元素为子Agent的key
        :return: None
        """

        tools_info: Optional[List[Tool]] = []

        if tools is not None:
            for tool in tools:
                self.tools[tool.__name__] = tool
                tools_info.append(standardize_tool(tool))

        self.agent_factory.register_agent(key, AgentInfo(
            description=description, prompt=prompt, tools=tools_info, sub_agents=sub_agents
        ))

    def create_core(self, key: Union[Tuple[str, str], List[Tuple[Tuple[Tuple[str, str], ...], Message]]],
                    history_queue: Queue,
                    raw_response_queue: Optional[Queue[RequestResponsePair]] = None,
                    api_key: Optional[str] = None
                    ) -> AgentCore:
        """
        初始化引擎，指定入口Agent和消息记录队列。

        支持两种模式：

        - 正常启动：传入一个元组 (namespace, name) 作为入口Agent的key。
        - 断点恢复：传入一个历史消息列表，每个元素为 (key_chain, message)，按时间顺序排列。
          引擎会根据消息中的key_chain重建调用栈和各Agent的消息历史。

        :param key: 启动方式。

        - 若为元组 (namespace, name)，表示正常启动，该元组为入口 Agent 的 key。
        - 若为列表，表示断点恢复模式。列表元素为 (key_chain, message)，按时间顺序排列。传入空列表时，会抛出 ValueError。
        - 注意：恢复模式下传入的历史消息列表可能会被修改（恢复时会补全缺失消息以确保符合API规范）

        :param history_queue: 消息输出队列，运行期间产生的所有消息（附带调用链信息）都会被放入此队列
        :param raw_response_queue: 可选，用于输出原始请求/响应对的队列
        :param api_key: DeepSeek API密钥，默认从环境变量DEEPSEEK_API_KEY读取
        :return: 已初始化的AgentCore实例
        """
        # 根据key的类型确认工作模式为初始化或恢复
        if isinstance(key, tuple):
            return AgentCore(self.agent_factory.build((key,)), history_queue, api_key=api_key, raw_response_queue=raw_response_queue)
        else:
            if len(key) < 1:
                raise ValueError("历史记录列表不能为空")
            # 第一条记录，第一个字段
            # key: List[Tuple[Tuple[Tuple[str, str], ...], Message]]
            key_chain: Tuple[Tuple[str, str], ...] = key[0][0]
            core: AgentCore = AgentCore(self.agent_factory.build(key_chain),
                                        history_queue, api_key=api_key, raw_response_queue=raw_response_queue)

            # 获取最后一条消息和所有者
            # key: List[Tuple[Tuple[Tuple[str, str], ...], Message]]
            last_message: Message = key[-1][1]
            last_key_chain: Tuple[Tuple[str, str], ...] = key[-1][0]

            # 最后一条是UserMessage：不用管
            if isinstance(last_message, UserMessage):
                pass

            # 最后一条是SystemMessage：说明任务信息丢失，可以直接结束
            elif isinstance(last_message, SystemMessage):
                # 判断是否在入口
                recovery_msg: UserMessage = UserMessage(
                    content="复述：“由于进程被中断，未能接收到任务。请重新指派任务”"
                )
                key.append((last_key_chain, recovery_msg))
                core.history_queue.put((last_key_chain, recovery_msg))

            # 最后一条是AssistantMessage，
            elif isinstance(last_message, AssistantMessage):
                if last_message.tool_calls is not None:
                    tools_queue: Queue[ToolCall] = parse_tool_calls(last_message.tool_calls)
                    while tools_queue.qsize() > 0:
                        recovery_msg: ToolMessage = ToolMessage(
                            content="工具执行被中断",
                            tool_call_id=tools_queue.get().id
                        )
                        key.append((last_key_chain, recovery_msg))
                        core.history_queue.put((last_key_chain, recovery_msg))
                    # 添加系统提示
                    key.append((last_key_chain, SystemMessage(content=recovery_prompt)))
                    core.history_queue.put((last_key_chain, SystemMessage(content=recovery_prompt)))


            # 最后一条是ToolMessage：倒序遍历至当前agent的AssistantMessage
            elif isinstance(last_message, ToolMessage):
                # 倒走遍历至AssistanMessage
                last_messages: List[Tuple[Tuple[Tuple[str, str], ...], Message]] = []
                for snapshot in reversed(key):
                    # 确保调用链信息一致
                    if snapshot[0] == last_key_chain:
                        if isinstance(snapshot[1], AssistantMessage):
                            last_messages.append(snapshot)
                            break
                        last_messages.append(snapshot)

                last_assistant_message: Tuple[Tuple[Tuple[str, str], ...], AssistantMessage] = last_messages.pop()
                last_tool_messages: List[Tuple[Tuple[Tuple[str, str], ...], ToolMessage]] = [
                    last_tool_message for last_tool_message in reversed(last_messages)
                ]

                completed_count: int = len(last_tool_messages)
                tools_queue: Queue[ToolCall] = parse_tool_calls(last_assistant_message[1].tool_calls)

                # 调用的工具比完成的多，就触发补全
                if tools_queue.qsize() > completed_count:
                    while completed_count > 0:
                        tools_queue.get_nowait()
                        completed_count -= 1

                    while tools_queue.qsize() > 0:
                        recovery_msg: ToolMessage = ToolMessage(
                            content="工具执行被中断",
                            tool_call_id=tools_queue.get().id
                        )
                        key.append((last_key_chain, recovery_msg))
                        core.history_queue.put((last_key_chain, recovery_msg))

                    # 添加系统提示
                    key.append((last_key_chain, SystemMessage(content=recovery_prompt)))
                    core.history_queue.put((last_key_chain, SystemMessage(content=recovery_prompt)))

                elif tools_queue.qsize() < completed_count:
                    raise ImportError("数据损坏，数据显示已执行工具多于调用得工具")

            # 消息已补全，正式开始恢复
            for snapshot in key:
                current_key_chain: Tuple[Tuple[str, str], ...] = snapshot[0]
                current_message: Message = snapshot[1]

                # 所有权标记未改变
                if current_key_chain == key_chain:
                    # 直接写入消息
                    core.agent.messages.append(current_message)

                # 所有权转移至父agent
                elif len(current_key_chain) < len(key_chain):
                    # 更新调用链
                    key_chain = current_key_chain
                    # 返回父agent写入消息
                    core.back()
                    core.agent.messages.append(current_message)

                # 所有权转移至子agent
                else:
                    # 更新调用链
                    key_chain = current_key_chain
                    # 创建子agent写入消息
                    sub_agent: Agent = self.agent_factory.build(key_chain)
                    core.sub_agent(sub_agent)
                    core.agent.messages.append(current_message)


            return core

    def run(self, input_queue: Queue[str],
            key: Union[Tuple[str, str], List[Tuple[Tuple[Tuple[str, str], ...], Message]]],
            history_queue: Queue,
            raw_response_queue: Optional[Queue[RequestResponsePair]] = None,
            api_key: Optional[str] = None
            ):
        """
        开始任务，阻塞直到所有Agent完成工作。
        此方法会：

        1. 调用 create_core 创建 AgentCore 实例（支持正常启动或历史恢复）。
        2. 正常启动时，从 input_queue 获取初始任务；恢复时，直接从历史中恢复执行状态，不再读取初始任务。恢复时，传入循环函数的 task 参数为 None，会被 init 函数自动忽略。
        3. 反复调用 main 函数处理消息循环，直到调用栈为空

        注意：在恢复模式下，input_queue 仍然用于在 ReactiveReAct 循环中接收新的用户输入。

        :param input_queue: 用户输入队列，用于接收新消息（在ReactiveReAct模式下需要）
        :param key: 同create_core的key参数，用于确定启动方式
        :param history_queue: 消息输出队列，同create_core
        :param raw_response_queue: 可选，原始请求/响应队列
        :param api_key: DeepSeek API密钥
        :return: None
        """
        core: AgentCore = self.create_core(key=key, history_queue=history_queue, raw_response_queue=raw_response_queue, api_key=api_key)
        depth: int = len(core.stack) + 1

        task: Optional[str]
        if not core.agent.messages:
            task = input_queue.get(block=True)
        else:
            task = None
        while depth > 0:
            return_value: Optional[str] = main(core, self.agent_factory, task, self.tools, input_queue)

            # 有返回值说明调用了子Agent，返回值是子Agent的任务
            if return_value is not None:
                task = return_value
                depth += 1
            else:
                task = None
                depth += -1