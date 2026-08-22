"""迷你 LangGraph：用 ~180 行 Python 复刻状态图引擎的核心机制。

覆盖四个概念：
- 状态（State）：带 reducer 的共享通道，节点只返回增量
- 节点（Node）：纯函数，读全量状态、写增量
- 边（Edge）：静态边 + 条件边，决定下一个超步激活谁
- 超步（Superstep）：一轮「激活 -> 执行 -> 合并」的原子推进

零依赖，Python 3.10+，直接运行可看到逐超步的执行轨迹。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

END = "__end__"


@dataclass
class Channel:
    """状态里的一个字段。reducer 决定多次写入如何合并；缺省为覆盖。"""
    default: Any = None
    reducer: Callable[[Any, Any], Any] | None = None


NodeFn = Callable[[dict], dict]
Router = Callable[[dict], str]


class StateGraph:
    def __init__(self, schema: dict[str, Channel]):
        self.schema = schema
        self.nodes: dict[str, NodeFn] = {}
        self.edges: dict[str, list[str]] = {}
        self.cond_edges: dict[str, tuple[Router, dict[str, str]]] = {}
        self.entry: str | None = None

    def add_node(self, name: str, fn: NodeFn) -> None:
        if name in self.nodes:
            raise ValueError(f"节点重复: {name}")
        self.nodes[name] = fn

    def add_edge(self, src: str, dst: str) -> None:
        self.edges.setdefault(src, []).append(dst)

    def add_conditional_edges(self, src: str, router: Router, mapping: dict[str, str]) -> None:
        self.cond_edges[src] = (router, mapping)

    def set_entry_point(self, name: str) -> None:
        self.entry = name

    def compile(self, checkpointer: "MemorySaver | None" = None) -> "CompiledGraph":
        if self.entry is None:
            raise ValueError("缺少入口节点")
        return CompiledGraph(self, checkpointer)


class MemorySaver:
    """按 thread 保存每个超步结束后的状态快照，对应 LangGraph 的 checkpointer。"""

    def __init__(self):
        self.store: dict[str, list[dict]] = {}

    def put(self, thread: str, snapshot: dict) -> None:
        self.store.setdefault(thread, []).append(dict(snapshot))

    def get(self, thread: str) -> list[dict]:
        return self.store.get(thread, [])


@dataclass
class CompiledGraph:
    graph: StateGraph
    checkpointer: MemorySaver | None = None
    max_supersteps: int = 25

    def _initial_state(self, inputs: dict) -> dict:
        state = {k: ch.default for k, ch in self.graph.schema.items()}
        for k, v in inputs.items():
            if k not in self.graph.schema:
                raise KeyError(f"输入字段 {k} 不在状态 schema 里")
            state[k] = v
        return state

    def _apply(self, state: dict, updates: dict) -> dict:
        """合并节点返回的增量：有 reducer 走合并，没有就覆盖。"""
        for k, v in updates.items():
            if k not in self.graph.schema:
                raise KeyError(f"节点写入了未声明的字段 {k}")
            reducer = self.graph.schema[k].reducer
            state[k] = reducer(state[k], v) if reducer else v
        return state

    def _next_nodes(self, current: str, state: dict) -> list[str]:
        targets: list[str] = []
        targets.extend(self.graph.edges.get(current, []))
        if current in self.graph.cond_edges:
            router, mapping = self.graph.cond_edges[current]
            key = router(state)
            if key not in mapping:
                raise ValueError(f"条件边 {current} 路由到了未知键: {key}")
            targets.append(mapping[key])
        return targets

    def invoke(self, inputs: dict, thread: str = "default", trace: bool = False) -> dict:
        state = self._initial_state(inputs)
        active = [self.graph.entry]
        for step in range(self.max_supersteps):
            updates: dict = {}
            for name in active:                       # 本超步所有激活节点
                result = self.graph.nodes[name](state)  # 读全量状态
                if result:
                    updates = {**updates, **result}     # 收集增量
                if trace:
                    print(f"  [超步 {step}] 节点 {name} 返回增量 {result}")
            state = self._apply(state, updates)         # 一次性合并，超步原子性
            if self.checkpointer:
                self.checkpointer.put(thread, state)
            nxt = [t for name in active for t in self._next_nodes(name, state) if t != END]
            if trace:
                print(f"  [超步 {step}] 合并后状态 keys={list(state)} -> 下一批 {nxt or '结束'}")
            if not nxt or all(t == END for t in nxt):
                return state
            active = nxt
        raise RuntimeError(f"超过 {self.max_supersteps} 个超步仍未到达 END，图里有死循环")


# ---------- 下面是一个真实跑起来的示例：写作 Agent ----------
# plan -> search -> write -> review ->(不满意回到 write，最多 2 轮)-> END

def plan(state: dict) -> dict:
    return {"plan": "拆解主题：先讲状态，再讲超步", "messages": ["plan: 生成大纲"]}


def search(state: dict) -> dict:
    return {"retrieved": ["BSP 模型论文", "LangGraph 官方文档"], "messages": ["search: 找到 2 份资料"]}


def write(state: dict) -> dict:
    round_no = len([m for m in state["messages"] if m.startswith("write:")]) + 1
    return {"draft": f"第 {round_no} 版草稿（基于 {len(state['retrieved'])} 份资料）",
            "messages": [f"write: 产出第 {round_no} 版"]}


def review(state: dict) -> dict:
    drafts = len([m for m in state["messages"] if m.startswith("write:")])
    verdict = "approved" if drafts >= 2 else "rewrite"
    return {"verdict": verdict, "messages": [f"review: {verdict}"]}


def route_after_review(state: dict) -> str:
    return state["verdict"]


def build_graph(checkpointer=None) -> CompiledGraph:
    g = StateGraph({
        "plan": Channel(default=""),
        "retrieved": Channel(default=[]),
        "draft": Channel(default=""),
        "verdict": Channel(default=""),
        "messages": Channel(default=[], reducer=lambda a, b: a + b),  # 追加式通道
    })
    g.add_node("plan", plan)
    g.add_node("search", search)
    g.add_node("write", write)
    g.add_node("review", review)
    g.set_entry_point("plan")
    g.add_edge("plan", "search")
    g.add_edge("search", "write")
    g.add_edge("write", "review")
    g.add_conditional_edges("review", route_after_review,
                            {"approved": END, "rewrite": "write"})
    return g.compile(checkpointer)


if __name__ == "__main__":
    saver = MemorySaver()
    app = build_graph(saver)
    print("=== 逐超步执行轨迹 ===")
    final = app.invoke({}, thread="demo", trace=True)
    print("\n=== 最终状态 ===")
    for k, v in final.items():
        print(f"{k}: {v}")
    print(f"\n=== checkpoint 数量: {len(saver.get('demo'))} ===")
    print("第一个 checkpoint:", saver.get("demo")[0])
