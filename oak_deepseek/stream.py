import json
from queue import Queue
from typing import Optional, Dict, Iterator, Tuple, List, Union
from threading import Thread, Condition

from requests import Response

def parse_stream(line: bytes) -> Optional[Dict]:
    """
    将单条 SSE 响应行解析为字典。

    :param line: 形如 b'data: {...}' 的原始字节流行
    :return: 解析后的字典；若遇到 `data: [DONE]` 则返回 None
    """
    data = line[6:].decode("utf-8")
    if data == "[DONE]":
        return None
    data = json.loads(data)
    return data


class Stream:
    """
    流式响应的处理器。

    负责从 HTTP 响应行迭代器中异步收集数据块，并提供按需获取内容增量、
    工具调用片段或合并完整响应的能力。

    :ivar chunks: 存储已解析的响应块（字典）列表
    :ivar _cond: 线程条件变量，用于协调生产者（后台线程）和消费者之间的同步
    :ivar _finished: 标志位，表示迭代器是否已消费完毕
    """
    def __init__(self, iterator: Union[Iterator, List]):
        """
        初始化 Stream 实例，启动后台线程消费迭代器。

        :param iterator: 可迭代对象，通常为 `response.iter_lines()` 返回的生成器，
                         每个元素是原始字节行（b'data: {...}'）或空行。
                         支持传入列表，主要用于测试。
        """
        self.chunks: List[Dict] = []
        self._cond: Condition = Condition()
        self._finished: bool = False
        self._error: Optional[Exception] = None

        def init():
            try:
                for chunk in iterator:
                    if chunk:
                        if isinstance(chunk, bytes):
                            chunk = parse_stream(chunk)
                            if chunk is None:
                                break
                        with self._cond:
                            self.chunks.append(chunk)
                            self._cond.notify_all()
            except Exception as e:
                self._error = e
            with self._cond:
                self._finished = True
                self._cond.notify_all()

        Thread(target=init, daemon=True).start()

    def _deltas(self) -> Iterator[Tuple[Dict, Dict]]:
        """
        内部生成器：按顺序产出 (原始块, delta 字段) 的二元组。

        delta 字段是从块中的 `choices[0].delta` 提取的字典。
        若某个块不包含该结构，delta 为空字典。

        :return: 迭代器，产生 (data, delta)
        """
        idx: int = 0
        while True:
            if self._error:
                raise self._error
            with self._cond:
                while idx >= len(self.chunks) and not self._finished:
                    self._cond.wait()
                if idx >= len(self.chunks) and self._finished:
                    break
                data: Dict = self.chunks[idx]
                delta: Dict = data.get('choices', [{}])[0].get('delta', {})
                idx += 1
            yield data, delta

    def build_tool_calls(self, tool_calls_queue: Queue) -> Iterator[Dict]:
        """
        从队列中消费工具调用片段，组装成完整的工具调用字典。

        该方法用于 `get_from_chunks` 和 `build_full_response` 中处理流式工具调用。
        当遇到新的工具调用索引（index）时，会产出上一个已完成的工具调用字典，
        最后产出最后一个工具调用。

        :param tool_calls_queue: 队列，元素为单个工具调用片段（delta 中的 tool_calls 列表项）。
                                 当队列收到 None 时表示输入结束。
        :return: 迭代器，产出完整构造的工具调用字典，结构如下：
                 {
                     "index": int,
                     "id": str,
                     "type": "function",
                     "function": {"name": str, "arguments": str}
                 }
        """
        tool_call: Dict = {"index": None, "id": None, "type": None, "function": {"name": None, "arguments": ""}}
        while True:
            chunk = tool_calls_queue.get(block=True)
            if chunk is None:
                break
            tmp_id: Optional[str] = chunk[0].get("id")
            if tmp_id:
                tmp_index: int = chunk[0].get("index")
                if tmp_index != 0:
                    yield tool_call
                    tool_call = {"index": None, "id": None, "type": None,
                                       "function": {"name": None, "arguments": ""}}

                tool_call["index"] = tmp_index
                tool_call["id"] = chunk[0].get("id")
                tool_call["type"] = chunk[0].get("type")
                tool_call['function']["name"] = chunk[0]['function'].get("name")

                tmp_arguments: str = chunk[0]['function'].get("arguments")
                if tmp_arguments:
                    tool_call['function']["arguments"] += tmp_arguments
            else:
                tmp_arguments: str = chunk[0]['function'].get("arguments")
                if tmp_arguments:
                    tool_call['function']["arguments"] += tmp_arguments
        yield tool_call

    def get_from_chunks(self):
        """
        遍历流式数据块，按类别产出内容。

        该方法是一个生成器，依次产出以下类型的值：
        - ("content", 内容片段)
        - ("reasoning_content", 推理内容片段)
        - 最后产出 ("tool_calls", 完整工具调用列表)

        :return: 生成器，每次产出一个二元组 (类别, 值)
        """
        tool_calls_queue: Queue = Queue()
        build_calls: bool = False
        tool_calls_iterator = None
        for data, delta in self._deltas():

            content: Optional[str] = delta.get('content')
            tool_calls: Optional[List[Dict]] = delta.get('tool_calls')
            reasoning_content: Optional[str] = delta.get('reasoning_content')

            if content:
                yield "content", content

            if tool_calls:
                if not build_calls:
                    tool_calls_iterator: Iterator = self.build_tool_calls(tool_calls_queue)
                    build_calls = True
                tool_calls_queue.put(tool_calls)

            if reasoning_content:
                yield "reasoning_content", reasoning_content

        tool_calls_queue.put(None)
        if tool_calls_iterator:
            full_tool_calls: List[Dict] = []
            for tool_call in tool_calls_iterator:
                full_tool_calls.append(tool_call)
            yield "tool_calls", full_tool_calls



    def build_full_response(self) -> Dict:
        """
        将流式数据块合并为一个完整的非流式响应字典。

        该方法会消费所有数据块，聚合 content、reasoning_content 和 tool_calls，
        并收集最终的 usage、id、created 等元数据。

        :return: 完整响应字典，结构与 DeepSeek API 非流式响应一致，例如：
                 {
                     "id": "...",
                     "object": "chat.completion",
                     "created": 1234567890,
                     "model": "deepseek-reasoner",
                     "system_fingerprint": "...",
                     "usage": {...},
                     "choices": [{
                         "index": 0,
                         "message": {
                             "content": "...",
                             "reasoning_content": "...",
                             "tool_calls": [...]
                         },
                         "logprobs": None,
                         "finish_reason": "stop"
                     }]
                 }
        """
        tool_calls_queue: Queue = Queue()
        build_calls: bool = False
        tool_calls_iterator = None
        parsed_response: Dict = {'choices': [
            {'index': None, 'message': {'content': '', 'tool_calls': None, 'reasoning_content': ''}, 'logprobs': None,
             'finish_reason': None}]}

        for data, delta in self._deltas():
            content: Optional[str] = delta.get('content')
            tool_calls: Optional[List[Dict]] = delta.get('tool_calls')
            reasoning_content: Optional[str] = delta.get('reasoning_content')

            if content:
                parsed_response['choices'][0]['message']['content'] += content

            if tool_calls:
                if not build_calls:
                    tool_calls_iterator: Iterator = self.build_tool_calls(tool_calls_queue)
                    build_calls = True
                tool_calls_queue.put(tool_calls)

            if reasoning_content:
                parsed_response['choices'][0]['message']['reasoning_content'] += reasoning_content

            if data.get('usage'):
                parsed_response['id'] = data.get('id')
                parsed_response['object'] = data.get('object')
                parsed_response['created'] = data.get('created')
                parsed_response['model'] = data.get('model')
                parsed_response['system_fingerprint'] = data.get('system_fingerprint')
                parsed_response['usage'] = data.get('usage')
                parsed_response['choices'][0]['index'] = data.get('choices')[0].get('index')
                parsed_response['choices'][0]['logprobs'] = data.get('choices')[0].get('logprobs')
                parsed_response['choices'][0]['finish_reason'] = data.get('choices')[0].get('finish_reason')

        tool_calls_queue.put(None)
        if tool_calls_iterator:
            full_tool_calls: List[Dict] = []
            for tool_call in tool_calls_iterator:
                full_tool_calls.append(tool_call)
            parsed_response['choices'][0]['message']['tool_calls'] = full_tool_calls

        return parsed_response