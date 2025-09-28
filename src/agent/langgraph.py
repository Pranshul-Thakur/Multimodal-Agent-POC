# src/agent/langgraph.py
import json
from typing import Dict, Any
from dataclasses import dataclass

@dataclass
class Node:
    id: str
    kind: str
    params: Dict[str, Any]

class LangGraph:
    def __init__(self, nodes: Dict[str, Node], edges: list, tools: dict):
        self.nodes = nodes
        self.edges = edges
        self.tools = tools
        self.state = {}

    @classmethod
    def from_json(cls, path: str, tools: dict):
        with open(path, "r", encoding="utf-8") as fh:
            j = json.load(fh)
        nodes = {n["id"]: Node(n["id"], n["kind"], n.get("params", {})) for n in j.get("nodes", [])}
        edges = j.get("edges", [])
        return cls(nodes, edges, tools)

    def run(self, start_node_id: str, input_data: dict):
        self.state[start_node_id] = input_data
        queue = [start_node_id]
        last_out = None
        while queue:
            nid = queue.pop(0)
            node = self.nodes.get(nid)
            if node is None:
                continue
            inp = self.state.get(nid, {})
            out = self._execute_node(node, inp)
            last_out = out
            for e in self.edges:
                if e.get("from") == nid:
                    self.state[e.get("to")] = out
                    queue.append(e.get("to"))
        return last_out

    def _execute_node(self, node: Node, inp: dict):
        kind = node.kind
        if kind == "planner":
            fn = self.tools.get("planner")
            if fn is None:
                return inp
            return fn(inp)
        if kind == "retriever":
            retr = self.tools.get("retriever")
            if retr is None:
                return {"docs": [], "query": inp.get("user_prompt")}
            query = inp.get("query") or inp.get("user_prompt")
            docs = retr.retrieve(query, k=node.params.get("k", 3))
            return {"docs": docs, "query": query}
        if kind == "generator":
            gen = self.tools.get("generator")
            if gen is None:
                return {"text": inp.get("query") or inp.get("user_prompt")}
            q = inp.get("query") or inp.get("user_prompt")
            contexts = inp.get("docs", [])
            text = gen.generate(q, contexts)
            return {"text": text}
        if kind == "tool":
            tool_name = node.params.get("tool_name")
            tool_fn = self.tools.get(tool_name)
            arg = inp.get("text") or inp.get("user_prompt") or ""
            if callable(tool_fn):
                try:
                    res = tool_fn(arg)
                except Exception:
                    res = None
                return {"result": res}
            return {"result": None}
        return inp

# ensure symbol is importable
__all__ = ["LangGraph", "Node"]