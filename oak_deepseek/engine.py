from queue import Queue
from typing import List, Callable, Literal, Optional, Tuple, Dict, Union
import os

from oak_deepseek.agent import AgentInfo, AgentFactory, Agent
from oak_deepseek.client import RequestResponsePair
from oak_deepseek.core import AgentCore
from oak_deepseek.loops import re_act, reactive_rea_ct
from oak_deepseek.models import Tool, Message
from oak_deepseek.tools import standardize_tool


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
                    api_key: str=os.getenv("DEEPSEEK_API_KEY")
                    ) -> AgentCore:
        """
        初始化引擎，指定入口Agent和消息记录队列。

        支持两种模式：

        - 正常启动：传入一个元组 (namespace, name) 作为入口Agent的key。
        - 断点恢复：传入一个历史消息列表，每个元素为 (agent_key, message)，按时间顺序排列。
          引擎会根据消息中的agent_key重建调用栈和各Agent的消息历史。

        :param key: 启动方式，元组表示正常启动，列表表示恢复模式
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
            api_key: str = os.getenv("DEEPSEEK_API_KEY")
            ):
        """
        开始任务，阻塞直到所有Agent完成工作。
        此方法会：
        1. 调用create_core创建AgentCore实例（支持正常启动或历史恢复）。
        2. 从input_queue获取初始任务（正常启动时）或继续已有任务（恢复时）。
        3. 根据当前Agent的循环模式，反复调用相应的循环函数，直到调用栈为空。

        :param input_queue: 用户输入队列，用于接收新消息（在ReactiveReAct模式下需要）
        :param key: 同create_core的key参数，用于确定启动方式
        :param history_queue: 消息输出队列，同create_core
        :param raw_response_queue: 可选，原始请求/响应队列
        :param api_key: DeepSeek API密钥
        :return: None
        """
        core: AgentCore = self.create_core(key=key, history_queue=history_queue, raw_response_queue=raw_response_queue, api_key=api_key)
        agent_count: int = len(core.stack) + 1
        task: str = input_queue.get(block=True)
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