from typing import Optional, List, Tuple, Dict
from pydantic import BaseModel

from oak_deepseek.models import Tool, Message
from oak_deepseek.tools import standardize_tool

def plan(steps: List[str]):
    """
    调用此工具时，不能调用其它工具
    在你开始执行任务前，需要先规划好步骤，并将步骤写入列表
    :param steps: 步骤列表，每个元素表示一个步骤
    """
    pass

def next_step():
    """
    调用此工具时，不能调用其它工具
    当你确认完成当前阶段任务时，调用此工具
    """
    pass

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
    description: str
    prompt: str
    tools: List[Tool]

    loop: str
    sub_agents: Optional[List[Tuple[str, str]]] = None

# agent自己维护好自己的消息记录
class Agent:
    def __init__(self, key: Tuple[str, str], info: AgentInfo):
        self.key: Tuple[str, str] = key
        self.info: AgentInfo = info
        self.messages: List[Message] = []

# 从元数据中组装agent实例
class AgentFactory:
    def __init__(self):
        self.agents: Dict[Tuple[str, str], AgentInfo] = {}

    def register_agent(self, key: Tuple[str, str], agent: AgentInfo):
        self.agents[key] = agent

    def build(self, key: Tuple[str, str]) -> Agent:
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