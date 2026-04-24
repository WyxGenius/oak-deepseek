import copy
from typing import Optional, List, Tuple, Dict, Literal
from pydantic import BaseModel

from oak_deepseek.types import Message, Tool
from oak_deepseek.tools import standardize_tool


def choose_agent(agent: Tuple[str, str], task: str) -> str:
    """
    此工具不能与任何工具**同时被调用**，包括 `choose_agent` 工具
    使用这个工具可以和助手对话，这是你与助手沟通的**唯一途径**。

    :param agent: (命名空间, 名字)
    :param task: AI Agent需要完成的任务，或者你要对它说的内容
    :return: AI Agent给你的回复
    """
    pass

# llm参数设置
class LLMConfig(BaseModel):
    """
    Agent的llm配置

    :ivar with_stream: 是否启用流式输出（默认 False）
    """
    model: Literal["deepseek-v4-flash", "deepseek-v4-pro"] = "deepseek-v4-flash"
    reasoning_effort: Literal["high", "max"] = "max"
    with_stream: bool = False

# agent元数据
class AgentInfo(BaseModel):
    """
    Agent的元数据，描述Agent的静态配置。

    :ivar description: Agent的简短描述
    :ivar prompt: 系统提示词
    :ivar tools: 可调用的工具列表
    :ivar sub_agents: 可选，可调用的子 Agent 列表，每个元素为 (namespace, name)。
                      **注意**：若该列表非空，框架会自动注入 `choose_agent` 工具，
                      并将子 Agent 信息（命名空间、名字、简介）拼接到提示词中。
                      若为空或 None，则不会添加 `choose_agent` 工具。
    :ivar rm_rf_memory: 是否在返回父 Agent 时递归清理记忆缓存。
                        若为 True，则在该 Agent 完成并调用 `back()` 时，
                        会删除 `AgentCore.memory` 中以当前 Agent 的 `KeyChain` 为前缀的所有条目
                        （包含自身及其子树），实现彻底的记忆清理。
                        默认 False。
    """

    description: str
    prompt: str
    tools: Optional[List[Tool]]
    with_stream: bool = False
    sub_agents: Optional[List[Tuple[str, str]]] = None
    rm_rf_memory: bool = False

    llm_config: LLMConfig = LLMConfig()

# agent自己维护好自己的消息记录
class Agent:
    """
    Agent实例，包含其元数据和消息历史。

    :ivar key_chain: 调用链元组，每个元素是 (namespace, name)，表示从根 Agent 到当前 Agent 的路径。
                     例如 `(('sys','root'), ('sys','calculator'))`。
    :ivar info: AgentInfo对象，包含静态配置
    :ivar messages: 该 Agent 的消息历史列表，按时间顺序排列。
                    消息会被 `AgentCore` 持久化到队列，并在恢复时重新加载。
    """
    def __init__(self, key_chain: Tuple[Tuple[str, str], ...], info: AgentInfo):
        """
        初始化Agent实例。

        :param key_chain: 一个元组，每个元素都是命名空间ID元组，表示调用链。
        :param info: Agent的元数据
        """
        self.key_chain: Tuple[Tuple[str, str], ...] = key_chain
        self.info: AgentInfo = info
        self.messages: List[Message] = []

# 从元数据中组装agent实例
class AgentFactory:
    """
    Agent工厂，负责注册和构建Agent实例。

    主要功能：
    - 通过 `register_agent` 注册静态配置（AgentInfo）。
    - 通过 `build` 根据 `key_chain` 动态创建 Agent 实例。
    - 如果 Agent 配置了子 Agent（sub_agents 非空），则自动：
        1. 将 `choose_agent` 工具加入工具列表。
        2. 将子 Agent 的描述信息（命名空间、名字、简介）拼接到提示词中。
    """
    def __init__(self):
        self.agents: Dict[Tuple[str, str], AgentInfo] = {}

    def register_agent(self, key: Tuple[str, str], agent: AgentInfo):
        """
        注册一个Agent的元数据。如果 key 已存在，则覆盖原有配置。

        :param key: Agent的唯一标识 (namespace, name)
        :param agent: Agent的元数据
        """
        self.agents[key] = agent

    def build(self, key_chain: Tuple[Tuple[str, str], ...]) -> Agent:
        """
        根据 key_chain 构建一个 Agent 实例。

        构建步骤：
        1. 验证 key_chain[-1] 是否已注册，否则抛出 KeyError。
        2. 深拷贝对应的 AgentInfo，防止修改原始配置。
        3. 创建 Agent 实例，初始消息列表为空。
        4. 如果 sub_agents 非空：
           - 将 `choose_agent` 工具添加到工具列表。
           - 遍历 sub_agents，验证每个子 Agent 已注册，并将它们的描述信息拼接到提示词末尾。

        注意：由于每次调用 `build` 都会深拷贝 `AgentInfo` 并拼接子 Agent 信息，
        因此构建出的 `Agent` 实例不应跨会话重复使用，以免提示词重复拼接。

        :param key_chain: 调用链元组，最后一个元素是要构建的 Agent 的 key。
        :return: 构建好的 Agent 实例（消息列表为空）
        :raises KeyError: 如果 key_chain[-1] 或任何 sub_agent 未注册
        """
        # 检查要构建的agent是否存在
        if self.agents.get(key_chain[-1]) is None:
            raise KeyError(f"agent {key_chain[-1]} 未注册")

        # 从表中查出agent信息，写入实例
        agent_info: AgentInfo = self.agents.get(key_chain[-1]).model_copy(deep=True)

        # key_chain不可变，调用方无需拷贝，Agent 内部会深拷贝
        agent: Agent = Agent(key_chain=key_chain, info=agent_info)

        # 将choose_agent工具加入工具列表，并将可用Agent信息拼接到提示词中
        keys: Optional[List[Tuple[str, str]]] = agent_info.sub_agents
        if keys:
            if agent.info.tools is None:
                agent.info.tools = []

            agent.info.tools.append(standardize_tool(choose_agent))
            prompt_about_sub_agent = "\n\n**你可以让这些AI Agent辅助你完成任务：**"

            for sub_agent in keys:
                if self.agents.get(sub_agent) is None:
                    raise KeyError(f"agent {sub_agent} 未注册")
                prompt_about_sub_agent += f"\n\n(namespace={sub_agent[0]}, name={sub_agent[1]})\n简介：{self.agents[sub_agent].description}"

            agent.info.prompt += prompt_about_sub_agent

        return agent