"""The agent as a LangGraph graph with three nodes: agent, tools and verify.

    START -> agent -> (asked for a tool?) -> tools -> agent -> ...
                   -> (final answer)      -> verify -> (every number grounded?) -> END
                                                    -> (not grounded) -> agent

Usage (from the project root):  python main.py "your question"
"""
import json
import re
import time

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from frog_agent.config import LOG_DIR, MAX_STEPS, MAX_TOKENS, MAX_VERIFY_RETRIES, MODEL, ROOT
from frog_agent.prompts import SYSTEM_PROMPT
from tools import TOOLS

load_dotenv(ROOT / ".env")


class AgentState(MessagesState):
    """The conversation (messages) plus what verify needs to remember between rounds."""
    verify_rejections: int
    unsupported_numbers: list[str]


# --- Grounding check: every number in the answer must appear in a tool result ---

NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")


def numbers_in(text):
    """Numbers written in the text, as strings (commas removed: 12,910 -> 12910)."""
    return [n.replace(",", "") for n in NUMBER.findall(str(text)) if n.replace(",", "").strip(".")]


def is_grounded(number, sources):
    """True if the number appears in a source, allowing rounding (12.7 matches 12.712457)
    and percentages (42.0 matches 0.42)."""
    decimals = len(number.split(".")[1]) if "." in number else 0
    value = float(number)
    tolerance = 0.5 * 10 ** -decimals + 1e-9
    return any(abs(s - value) <= tolerance or abs(s * 100 - value) <= tolerance for s in sources)


def check_answer(answer, messages):
    """Return the numbers in the answer that no tool result (or the question / system prompt) contains."""
    # messages[0] is the question; later HumanMessages are verify's own feedback and don't count
    allowed = [SYSTEM_PROMPT, messages[0].content] + [m.content for m in messages if isinstance(m, ToolMessage)]
    sources = [float(n) for text in allowed for n in numbers_in(text)]
    # Small integers (list numbering, "2 nights", "3 frogs") are too common to check meaningfully
    to_check = [n for n in dict.fromkeys(numbers_in(answer)) if float(n) > 10 or "." in n]
    return [n for n in to_check if not is_grounded(n, sources)]


# --- The graph ---

def build_graph(llm=None):
    """Build and compile the graph. Pass llm to swap the model (e.g. a fake one for tests)."""
    llm = (llm or ChatAnthropic(model=MODEL, max_tokens=MAX_TOKENS)).bind_tools(TOOLS)

    def agent(state: AgentState):
        # The system prompt is added on every call instead of being stored in the state
        response = llm.invoke([SystemMessage(SYSTEM_PROMPT)] + state["messages"])
        return {"messages": [response]}

    def verify(state: AgentState):
        answer = str(state["messages"][-1].text)
        unsupported = check_answer(answer, state["messages"])
        rejections = state.get("verify_rejections", 0)
        if not unsupported or rejections >= MAX_VERIFY_RETRIES:
            return {"unsupported_numbers": unsupported}
        feedback = (f"Verification failed: these numbers in your answer do not appear in any tool result: "
                    f"{', '.join(unsupported)}. Compute them with a tool, or remove them from the answer.")
        return {"messages": [HumanMessage(feedback)], "verify_rejections": rejections + 1,
                "unsupported_numbers": unsupported}

    def after_agent(state: AgentState):
        return "tools" if state["messages"][-1].tool_calls else "verify"

    def after_verify(state: AgentState):
        # verify added a feedback message only when the answer has to be fixed
        return "agent" if isinstance(state["messages"][-1], HumanMessage) else END

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent)
    # A failing tool returns its error as a normal result, so the model can fix itself next round
    graph.add_node("tools", ToolNode(TOOLS, handle_tool_errors=lambda e: f"ERROR: {type(e).__name__}: {e}"))
    graph.add_node("verify", verify)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", after_agent, ["tools", "verify"])
    graph.add_edge("tools", "agent")
    graph.add_conditional_edges("verify", after_verify, ["agent", END])
    return graph.compile()


def save_graph_picture(graph, path=ROOT / "outputs" / "graph.mmd"):
    """Save the graph as Mermaid text (paste it into mermaid.live, or see slides/make_figures.py)."""
    path.write_text(graph.get_graph().draw_mermaid(), encoding="utf-8")
    return path


def run_agent(question, verbose=True, graph=None):
    graph = graph or build_graph()
    # recursion_limit counts graph steps, not model calls: each tool round is two steps (agent + tools)
    # and each verify rejection adds two more (verify + agent)
    config = {"recursion_limit": 2 * MAX_STEPS + 2 * (MAX_VERIFY_RETRIES + 1)}
    trace = []
    calls = {}  # tool_call_id -> its entry in trace
    step = 0
    answer = None
    unsupported = []

    try:
        # stream_mode="updates" yields what each node added, as soon as it finishes
        for update in graph.stream({"messages": [HumanMessage(question)]}, config, stream_mode="updates"):
            for node, output in update.items():
                if node == "verify":
                    unsupported = output.get("unsupported_numbers", [])
                    # rejected: sent back to the agent; gave_up: still unsupported after MAX_VERIFY_RETRIES
                    status = "rejected" if output.get("messages") else "gave_up" if unsupported else "passed"
                    trace.append({"step": step, "verify": status, "unsupported": unsupported})
                    if verbose:
                        print(f"\n[verify] {status}" + (f": {', '.join(unsupported)}" if unsupported else ""))
                    continue
                for message in output["messages"]:
                    if node == "agent":
                        step += 1
                        usage = message.usage_metadata or {}
                        trace.append({"step": step, "tokens": usage.get("total_tokens", 0)})
                        for call in message.tool_calls:
                            calls[call["id"]] = {"step": step, "tool": call["name"], "input": call["args"]}
                            trace.append(calls[call["id"]])
                            if verbose:
                                args = json.dumps(call["args"], ensure_ascii=False)[:300]
                                print(f"\n[step {step}] {call['name']}: {args}")
                        if not message.tool_calls:
                            answer = str(message.text)
                    else:
                        # tool_call_id links each result to the request it answers
                        calls[message.tool_call_id]["output"] = message.content
                        if verbose:
                            print(str(message.content)[:400])
    except GraphRecursionError:
        answer = None
    if answer is None:
        answer = f"Stopped after {step} model calls without a final answer."
    elif unsupported:
        # verify gave up: keep the answer, but say plainly what could not be checked
        answer += f"\n\n(Note: these numbers could not be traced to a tool result: {', '.join(unsupported)})"
    trace.append({"step": step, "answer": answer})

    # Save the full trace: it is the debugging tool and material for the slides
    LOG_DIR.mkdir(exist_ok=True)
    log_path = LOG_DIR / f"run_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump({"question": question, "steps": trace}, f, ensure_ascii=False, indent=2)
    if verbose:
        print(f"\n[log] {log_path.relative_to(ROOT)}")
    return answer, trace
