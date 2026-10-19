"""Build every figure and code card for the agent slides. Free: no model calls.

Usage (from the project root):
    python slides/make_figures.py                         # every figure
    python slides/make_figures.py --trace logs/run_X.json # only a trace figure for that run
    python slides/make_figures.py --trace logs/run_X.json --answer "..."  # pick the answer line shown
Reads the real code, logs/ and outputs/eval_results.csv, so re-running it after a new
evaluation or a new demo run updates the slides. Everything is saved to slides/figures/.
"""
import argparse
import inspect
import json
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from langchain_anthropic.chat_models import convert_to_anthropic_tool
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from pygments import highlight
from pygments.formatters import ImageFormatter
from pygments.lexers import JsonLexer, PythonLexer

from frog_agent import graph as agent_graph
from frog_agent.config import LOG_DIR, MAX_ROWS, MAX_TOOL_OUTPUT, MAX_VERIFY_RETRIES, OUTPUT_DIR, PYTHON_TIMEOUT_S
from tools import TOOLS, run_sql

OUT = ROOT / "slides" / "figures"
CODE_OUT = ROOT / "slides" / "code"

# One small palette for all slides: neutral ink, one accent, and status colors that always come with a text label
INK, MUTED, GRID, SURFACE = "#1f2328", "#59636e", "#d1d9e0", "#ffffff"
ACCENT, ACCENT_SOFT = "#2f6fdf", "#e3edfc"
GOOD, GOOD_SOFT = "#1a7f37", "#dafbe1"
WARN, WARN_SOFT = "#9a6700", "#fff8c5"
BAD, BAD_SOFT = "#cf222e", "#ffebe9"

plt.rcParams.update({"font.family": ["Helvetica", "DejaVu Sans"], "font.size": 11, "text.color": INK,
                     "axes.edgecolor": GRID, "axes.labelcolor": MUTED,
                     "xtick.color": MUTED, "ytick.color": MUTED})


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print("saved", OUT / name)


# --- Slide 1: the graph ---

def figure_graph():
    """Draw the compiled LangGraph graph. Edges are read from the real graph, so the picture can't drift from the code."""
    edges = {(e.source, e.target) for e in agent_graph.build_graph(llm=_NoModel()).get_graph().edges}
    positions = {"__start__": (0.5, 4.2), "agent": (3.0, 4.2), "tools": (6.6, 4.2),
                 "verify": (3.0, 1.8), "__end__": (6.6, 1.8)}
    titles = {"__start__": "START", "agent": "agent", "tools": "tools", "verify": "verify", "__end__": "END"}
    subtitles = {"agent": f"Claude + {len(TOOLS)} tools", "tools": " · ".join(t.name for t in TOOLS),
                 "verify": "every number grounded?"}
    labels = {("agent", "tools"): "tool calls", ("tools", "agent"): "results, incl. errors",
              ("agent", "verify"): "final answer", ("verify", "agent"): "number with no source",
              ("verify", "__end__"): "all numbers grounded", ("__start__", "agent"): "question"}
    assert edges == set(labels), f"graph edges changed, update the labels: {edges ^ set(labels)}"

    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.set_xlim(-0.6, 8.2); ax.set_ylim(0.6, 5.4); ax.axis("off")
    sizes = {}
    for node, (x, y) in positions.items():
        terminal = node.startswith("__")
        w, h = (1.1, 0.6) if terminal else (2.7 if node == "tools" else 2.0, 0.95)
        sizes[node] = (w, h)
        face = "#f6f8fa" if terminal else (ACCENT_SOFT if node != "verify" else WARN_SOFT)
        edge = MUTED if terminal else (ACCENT if node != "verify" else WARN)
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.18",
                                    fc=face, ec=edge, lw=1.6))
        ax.text(x, y + (0.13 if node in subtitles else 0), titles[node], ha="center", va="center",
                fontsize=15 if not terminal else 11, weight="bold", color=INK)
        if node in subtitles:
            ax.text(x, y - 0.22, subtitles[node], ha="center", va="center", fontsize=8.5, color=MUTED)

    def anchor(node, toward):
        (x, y), (w, h) = positions[node], sizes[node]
        tx, ty = positions[toward]
        if abs(tx - x) > abs(ty - y):
            return (x + (w / 2 if tx > x else -w / 2), y)
        return (x, y + (h / 2 if ty > y else -h / 2))

    for (a, b), label in labels.items():
        start, end = anchor(a, b), anchor(b, a)
        pair = (b, a) in labels  # two-way edges are drawn as two curves
        bend = 0.35 if pair else 0.0
        offset = {("agent", "tools"): (0, -0.42), ("tools", "agent"): (0, 0.42),
                  ("agent", "verify"): (-0.95, 0), ("verify", "agent"): (0.95, 0)}.get((a, b), (0, 0.18))
        ax.add_patch(FancyArrowPatch(start, end, connectionstyle=f"arc3,rad={bend}", arrowstyle="-|>",
                                     mutation_scale=16, lw=1.5, color=MUTED, shrinkA=2, shrinkB=2))
        mx, my = (start[0] + end[0]) / 2 + offset[0], (start[1] + end[1]) / 2 + offset[1]
        ax.text(mx, my, label, ha="center", va="center", fontsize=9, color=MUTED,
                bbox=dict(fc=SURFACE, ec="none", pad=1.5))
    ax.text(-0.5, 5.25, "The agent as a LangGraph graph", fontsize=14, weight="bold", va="top")
    ax.text(-0.5, 0.75, f"Limits: recursion_limit stops endless loops · verify sends an answer back at most {MAX_VERIFY_RETRIES} times",
            fontsize=8.5, color=MUTED)
    save(fig, "slide1_graph.png")


class _NoModel:
    """Stand-in model so the graph can be built without an API key (only its shape is needed here)."""
    def bind_tools(self, tools):
        return self


# --- Code cards (slides 2 and 3) ---

def code_card(code, name, title, lexer=None):
    """Render code as a dark card with syntax colors (Pygments), and save the plain text next to it."""
    code = textwrap.dedent(code).strip("\n")
    CODE_OUT.mkdir(parents=True, exist_ok=True)
    (CODE_OUT / name.replace(".png", ".txt")).write_text(code + "\n", encoding="utf-8")
    comment = "//" if isinstance(lexer, JsonLexer) else "#"
    image = highlight(f"{comment} {title}\n\n{code}\n", lexer or PythonLexer(),
                      ImageFormatter(font_name="Menlo", font_size=26, style="github-dark",
                                     line_numbers=False, image_pad=36, line_pad=6))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_bytes(image)
    print("saved", OUT / name)


def source_between(func, first, last):
    """The lines of func's source from the line containing `first` to the line containing `last`."""
    lines = inspect.getsource(func).splitlines()
    i = next(n for n, l in enumerate(lines) if first in l)
    j = next(n for n, l in enumerate(lines) if last in l and n >= i)
    return "\n".join(lines[i:j + 1])


def figure_code_cards():
    import tools
    code_card(source_between(agent_graph.build_graph, "graph = StateGraph", "return graph.compile()"),
              "slide2_build_graph.png", "frog_agent/graph.py: building the graph")
    tool_src = inspect.getsource(tools)
    sql_src = tool_src[tool_src.index("class SqlInput"):tool_src.index("class PythonInput")].strip()
    sql_src += "\n\n" + tool_src[tool_src.index("@tool(args_schema=SqlInput)"):tool_src.index("    try:", tool_src.index("@tool(args_schema=SqlInput)"))].rstrip()
    code_card(sql_src + "\n    ...", "slide2_run_sql_tool.png", "tools.py: a tool = a Pydantic schema + a docstring")
    code_card(json.dumps(convert_to_anthropic_tool(run_sql), indent=2), "slide2_run_sql_schema.png",
              "What the model actually receives for run_sql (generated from the code above)", JsonLexer())
    verify_src = source_between(agent_graph.build_graph, "def verify", 'return {"messages": [HumanMessage(feedback)]')
    code_card(verify_src + '\n                "unsupported_numbers": unsupported}', "slide3_verify_node.png",
              "frog_agent/graph.py: the verify node")


# --- Slide 3: a real trace, and the safeguards ---

def pick_trace():
    """Best demo for slide 3: a run with a verify rejection (else a self-correction), the shortest one, newest first."""
    logs = sorted(LOG_DIR.glob("run_*.json"), reverse=True)
    def score(path):
        steps = json.loads(path.read_text(encoding="utf-8"))["steps"]
        errors = any("ERROR" in str(s.get("output", "")) for s in steps)
        rejected = any(s.get("verify") == "rejected" for s in steps)
        return (2 * rejected + errors, -len(steps))
    return max(logs, key=score)  # max keeps the first (newest) among equals


def figure_trace(path=None, name="slide3_trace.png", answer=None):
    path = path or pick_trace()
    run = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for s in run["steps"]:
        if "tool" in s:
            arg = s["input"].get("query") or s["input"].get("code") or ""
            arg = " ".join(str(arg).split())
            out = str(s.get("output", ""))
            error = "ERROR" in out
            result = next((l for l in out.splitlines() if "ERROR" in l), out.splitlines()[0] if out else "") if error else \
                " ".join(out.split())
            rows.append((s["step"], s["tool"], arg, result, "error" if error else "tool"))
        elif "verify" in s:
            text = {"passed": "all numbers found in tool results",
                    "rejected": "sent back: no source for " + ", ".join(s["unsupported"]),
                    "gave_up": "still unsupported: " + ", ".join(s["unsupported"])}[s["verify"]]
            rows.append((s["step"], "verify", "", text, "rejected" if s["verify"] != "passed" else "passed"))
        elif "answer" in s:
            # answer: a line chosen from the answer for the slide (by default its first line)
            first = answer or next((l for l in s["answer"].splitlines() if l.strip()), "")
            rows.append((s["step"], "answer", "", first.replace("**", ""), "answer"))

    styles = {"tool": (SURFACE, INK, None), "error": (BAD_SOFT, BAD, "error → self-correction"),
              "rejected": (WARN_SOFT, WARN, "sent back"), "passed": (GOOD_SOFT, GOOD, "passed"),
              "answer": ("#f6f8fa", INK, None)}
    fig, ax = plt.subplots(figsize=(12, 0.9 + 0.5 * len(rows)))
    ax.set_xlim(0, 12); ax.set_ylim(len(rows) + 0.4, -1.3); ax.axis("off")
    ax.text(0, -1.0, "A real run, step by step", fontsize=14, weight="bold")
    ax.text(0, -0.45, f"Q: {run['question']}", fontsize=10.5, color=MUTED)
    for i, (step, tool, arg, result, kind) in enumerate(rows):
        face, color, tag = styles[kind]
        ax.add_patch(FancyBboxPatch((0, i + 0.08), 12, 0.84, boxstyle="round,pad=0,rounding_size=0.08",
                                    fc=face, ec=GRID, lw=0.8))
        ax.text(0.15, i + 0.5, str(step), va="center", fontsize=10, color=MUTED)
        ax.text(0.55, i + (0.36 if tag else 0.5), tool, va="center", fontsize=10.5, weight="bold", color=color)
        if arg:
            ax.text(2.0, i + 0.32, _clip(arg, 92), va="center", fontsize=8.5, family="Menlo", color=INK)
            ax.text(2.0, i + 0.7, "→ " + _clip(result, 110), va="center", fontsize=8.5, color=color if kind == "error" else MUTED)
        else:
            ax.text(2.0, i + 0.5, _clip(result, 105), va="center", fontsize=9.5, color=color)
        if tag:
            ax.text(0.55, i + 0.72, tag, va="center", fontsize=8, color=color)
    save(fig, name)
    print("   trace from", path.name)


def _clip(text, n):
    return text if len(text) <= n else text[:n - 1] + "…"


def figure_safeguards():
    items = [("Read-only database", "SQLite opened with mode=ro: the agent can't change the data"),
             ("Isolated process + timeout", f"run_python runs in its own Python process, stopped after {PYTHON_TIMEOUT_S} s"),
             ("Errors are results", "a failing tool returns ERROR text, the model reads it and corrects itself"),
             ("Grounding check (verify)", "every number in the answer must appear in a tool result"),
             ("Step limit", "LangGraph recursion_limit ends runs that never finish"),
             ("Bounded output", f"at most {MAX_ROWS} rows and {MAX_TOOL_OUTPUT:,} characters per tool result"),
             ("Full trace", "every run is saved to logs/ as JSON")]
    fig, ax = plt.subplots(figsize=(7.5, 0.6 + 0.55 * len(items)))
    ax.set_xlim(0, 10); ax.set_ylim(len(items) + 0.2, -0.9); ax.axis("off")
    ax.text(0, -0.5, "Safeguards", fontsize=14, weight="bold")
    for i, (name, detail) in enumerate(items):
        ax.text(0, i + 0.3, name, fontsize=11, weight="bold", color=ACCENT)
        ax.text(0, i + 0.68, detail, fontsize=9.5, color=MUTED)
    save(fig, "slide3_safeguards.png")


# --- Slide 4: evaluation ---

CATEGORIES = ["lookup", "aggregation", "data_quality", "trap", "thesis", "combined"]


def figure_eval():
    path = OUTPUT_DIR / "eval_results.csv"
    if not path.exists():
        print("skip slide 4: run python evaluate.py first"); return
    df = pd.read_csv(path)
    has_new = {"first_try_ok", "tokens", "verify_rejections"} <= set(df.columns)

    # Accuracy by category: one series, so one color and direct labels
    summary = df.groupby("category")["passed"].agg(["mean", "count"])
    summary = summary.reindex([c for c in CATEGORIES if c in summary.index])
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    bars = ax.bar(summary.index, summary["mean"] * 100, color=ACCENT, width=0.55)
    for bar, (_, r) in zip(bars, summary.iterrows()):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2, f"{r['mean']:.0%}  (n={int(r['count'])})",
                ha="center", fontsize=9.5, color=INK)
    ax.set_ylim(0, 112); ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"Accuracy by question type · overall {df['passed'].mean():.0%} ({df['passed'].sum()}/{len(df)})",
                 loc="left", fontsize=12, weight="bold")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.yaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
    save(fig, "slide4_accuracy.png")

    # The metrics table
    rows = []
    for cat in summary.index:
        part = df[df["category"] == cat]
        rows.append(_metric_row(cat, part, has_new))
    rows.append(_metric_row("All", df, has_new))
    header = ["Question type", "n", "Accuracy", "Tool calls / q", "First-try OK", "Tokens / q", "Verify rejections"]
    fig, ax = plt.subplots(figsize=(10, 0.4 * (len(rows) + 1)))
    ax.axis("off")
    table = ax.table(cellText=rows, colLabels=header, loc="center", cellLoc="center", colLoc="center")
    table.auto_set_font_size(False); table.set_fontsize(10); table.scale(1, 1.6)
    table.auto_set_column_width(range(len(header)))
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor(GRID)
        if r == 0:
            cell.set_text_props(weight="bold", color=INK); cell.set_facecolor("#f6f8fa")
        if r == len(rows):
            cell.set_text_props(weight="bold")
    title = "Evaluation" + ("" if has_new else "  (old run: re-run evaluate.py for the new metrics)")
    ax.set_title(title, loc="left", fontsize=12, weight="bold")
    save(fig, "slide4_metrics_table.png")


def _metric_row(name, part, has_new):
    errors, calls = part["tool_errors"].sum(), part["tool_calls"].sum()
    return [name, len(part), f"{part['passed'].mean():.0%}", f"{part['tool_calls'].mean():.1f}",
            f"{1 - errors / calls:.0%}" if calls else "—",
            f"{part['tokens'].mean():,.0f}" if has_new else "—",
            f"{int(part['verify_rejections'].sum())}" if has_new else "—"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", nargs="+", type=Path, help="log file(s) to draw as trace figures")
    parser.add_argument("--answer", help="the answer line to show in the trace figure (default: its first line)")
    args = parser.parse_args()
    if args.trace:
        # One figure per log, named after it: logs/run_20261017_120000.json -> trace_run_20261017_120000.png
        for log in args.trace:
            figure_trace(log, f"trace_{log.stem}.png", args.answer)
        sys.exit()
    figure_graph()
    figure_code_cards()
    figure_trace()
    figure_safeguards()
    figure_eval()
