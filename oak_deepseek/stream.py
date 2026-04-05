import json
from queue import Queue
from typing import Optional, Dict, Iterator, Tuple, List, Union
from threading import Thread, Condition

from requests import Response

def parse_stream(line: bytes) -> Optional[Dict]:
    """
    将单条响应流解析为字典
    """
    data = line[6:].decode("utf-8")
    if data == "[DONE]":
        return None
    data = json.loads(data)
    return data


class Stream:
    def __init__(self, iterator: Union[Iterator, List]):
        self.chunks: List[Dict] = []
        self._cond: Condition = Condition()
        self._finished: bool = False

        def init():
            for chunk in iterator:
                if chunk:
                    if isinstance(chunk, bytes):
                        chunk = parse_stream(chunk)
                        if chunk is None:
                            break
                    with self._cond:
                        self.chunks.append(chunk)
                        self._cond.notify_all()
            with self._cond:
                self._finished = True
                self._cond.notify_all()

        Thread(target=init, daemon=True).start()

    def _deltas(self) -> Iterator[Tuple[Dict, Dict]]:
        idx: int = 0
        while True:
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
        tool_calls_queue: Queue = Queue()
        build_calls: bool = False
        tool_calls_iterator = None
        parsed_response: Dict = {'choices': [
            {'index': None, 'message': {'content': '', 'tool_calls': None, 'reasoning_content': ''}, 'logprobs': None,
             'finish_reason': None}]}

        for data, delta in self._deltas():
            content: Optional[str] = delta.get('content')
            tool_calls: Optional[str] = delta.get('tool_calls')
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
            parsed_response['choices'][0]['message']['tool_calls']: List[Dict] = full_tool_calls

        return parsed_response