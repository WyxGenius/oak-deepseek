from queue import Queue
from typing import List, Callable, Literal, Optional, Tuple, Dict
import os

from oak_deepseek.agent import AgentInfo, AgentFactory
from oak_deepseek.client import RequestResponsePair
from oak_deepseek.engine import AgentEngine
from oak_deepseek.loops.strategies import re_act, reactive_enter
from oak_deepseek.models import Tool
from oak_deepseek.tool import standardize_tool


class AgentSession:
    def __init__(self):
        self.agent_factory: AgentFactory = AgentFactory()
        self.tools: Dict[str, Callable] = {}
        self.engine: Optional[AgentEngine] = None

    # 快速注册
    def create_agent(self,
                     key: Tuple[str, str],
                     description: str,
                     prompt: str,
                     tools: Optional[List[Callable]]=None,
                     loop: Literal["reactive_enter", "ReAct"]="ReAct",
                     sub_agents: Optional[List[Tuple[str, str]]]=None):
        """
        这个函数用来注册一个Agent至系统
        :param key: Tuple[str, str]，命名空间+名字
        :param description: str，Agent描述
        :param prompt: str，Agent提示词
        :param tools: Optional[List[Callable]]，可用工具
        :param loop: str，消息循环模式
        :param sub_agents: 可以调用的子Agent
        :return:
        """

        tools_info: Optional[List[Tool]] = []

        if tools is not None:
            for tool in tools:
                self.tools[tool.__name__] = tool
                tools_info.append(standardize_tool(tool))

        self.agent_factory.register_agent(key, AgentInfo(
            description=description, prompt=prompt, tools=tools_info, loop=loop, sub_agents=sub_agents
        ))

    def init_engine(self, key: Tuple[str, str],
                    history_queue: Queue,
                    raw_response_queue: Optional[Queue[RequestResponsePair]] = None,
                    api_key: str=os.getenv("DEEPSEEK_API_KEY")
                    ):
        """
        初始化引擎，指定入口Agent和消息记录队列
        :param key: Tuple[str, str]，入口agent的命名空间id
        :param history_queue: Queue，这个队列用会写入完整的消息记录，可以用于持久化等场景
        :param raw_response_queue: Queue[namedtuple("RequestResponsePair", ["request", "response"]]，记录原始请求与响应
        :param api_key: deepseek api key，不写的话默认从环境变量DEEPSEEK_API_KEY里取
        :return:
        """
        self.engine: AgentEngine = AgentEngine(self.agent_factory.build(key), history_queue,
                                               api_key=api_key, raw_response_queue=raw_response_queue)


    def run(self, input_queue: Queue[str]):
        """
        开始任务，指定接受用户消息的队列
        :param input_queue: Queue[str]，队列，用于接收用户消息
        :return:
        """
        agent_count: int = 1
        task: str = input_queue.get(block=True)
        while agent_count > 0:
            return_value = None
            # 检查当前Agent的工作模式
            if self.engine.agent.info.loop == "ReAct":
                return_value: Optional[str] = re_act(self.engine, self.agent_factory, task, self.tools)
            if self.engine.agent.info.loop == "reactive_enter":
                return_value: Optional[str] = reactive_enter(self.engine, self.agent_factory, task, self.tools, input_queue)

            # 有返回值说明调用了子Agent，返回值是子Agent的任务
            if return_value is not None:
                task = return_value
                agent_count += 1
            else:
                agent_count += -1