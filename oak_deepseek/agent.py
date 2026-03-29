from typing import Optional, List, Tuple, Dict
from pydantic import BaseModel

from oak_deepseek.models import Tool, Message
from oak_deepseek.tools import standardize_tool


def finished(conclusion: str):
    """
    调用此工具时，不能调用其它工具。
    当你要做最终总结时，不能直接发送总结内容，而是写在这个工具的参数里

    :param conclusion: 你的总结
    """
    pass

def choose_agent(agent: Tuple[str, str], task) -> str:
    """
    调用此工具时，不能调用其它工具。
    在某些时候，你需要使用这个工具给新的AI Agent布置任务

    :param agent: (命名空间, 名字)
    :param task: AI Agent需要完成的任务
    :return: AI Agent完成任务时，会返回你执行摘要
    """
    pass

# agent元数据
class AgentInfo(BaseModel):
    """
    Agent的元数据，描述Agent的静态配置。

    :ivar description: Agent的简短描述
    :ivar prompt: 系统提示词
    :ivar tools: 可调用的工具列表
    :ivar loop: 循环模式，如 "ReAct" 或 "ReactiveReAct"
    :ivar sub_agents: 可选，可调用的子Agent列表，每个元素为(namespace, name)
    """
    description: str
    prompt: str
    tools: List[Tool]

    loop: str
    sub_agents: Optional[List[Tuple[str, str]]] = None

# agent自己维护好自己的消息记录
class Agent:
    """
    Agent实例，包含其元数据和消息历史。

    :ivar key: 唯一标识 (namespace, name)
    :ivar info: AgentInfo对象
    :ivar messages: 该Agent的消息历史列表
    """
    def __init__(self, key: Tuple[str, str], info: AgentInfo):
        """
        初始化Agent实例。

        :param key: Agent的唯一标识 (namespace, name)
        :param info: Agent的元数据
        """
        self.key: Tuple[str, str] = key
        self.info: AgentInfo = info
        self.messages: List[Message] = []

# 从元数据中组装agent实例
class AgentFactory:
    """
    Agent工厂，负责注册和构建Agent实例。
    """
    def __init__(self):
        self.agents: Dict[Tuple[str, str], AgentInfo] = {}

    def register_agent(self, key: Tuple[str, str], agent: AgentInfo):
        """
        注册一个Agent的元数据。

        :param key: Agent的唯一标识
        :param agent: Agent的元数据
        """
        self.agents[key] = agent

    def build(self, key: Tuple[str, str]) -> Agent:
        """
        根据key构建一个Agent实例。
        自动添加finished工具，如有子Agent则添加choose_agent工具并拼接提示词。

        :param key: Agent的唯一标识
        :return: 构建好的Agent实例
        :raises KeyError: 如果key未注册
        """
        # 检查要构建的agent是否存在
        if self.agents.get(key) is None:
            raise KeyError(f"agent {key} 未注册")

        # 从表中查出agent信息，写入实例
        agent_info: AgentInfo = self.agents.get(key).model_copy(deep=True)
        agent: Agent = Agent(key=key, info=agent_info)

        # 将finished工具添加至列表
        agent.info.tools.append(standardize_tool(finished))

        # 将choose_agent工具加入工具列表，并将可用Agent信息拼接到提示词中
        keys: Optional[Tuple[str, str]] = agent_info.sub_agents
        if keys is not None:
            agent.info.tools.append(standardize_tool(choose_agent))
            prompt_about_sub_agent = "\n\n**你可以让这些AI Agent辅助你完成任务：**"
            for sub_agent in agent_info.sub_agents:
                prompt_about_sub_agent += f"\n\n(namespace={sub_agent[0]}, name={sub_agent[1]})\n简介：{self.agents[sub_agent].description}"
            agent.info.prompt += prompt_about_sub_agent

        return agent