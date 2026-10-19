import json
import os
import sys
import time

import anthropic
from dotenv import load_dotenv

from tools import get_schema, run_python, run_sql

load_dotenv()
client = anthropic.Anthropic()
MODEL = "claude-sonnet-5-5"
MAX_STEPS = 15  # stop the loop even if the agent never finishes

SYSTEM_PROMPT = """You are a research assistant for a dataset of tree-frog (Hyla savignyi) chorus recordings.
Answer questions by querying the data with your tools.

About the data (table "transitions"):
- Each row is a call sequence: frog i called, then frog j, then frog k (the current call).
  A row exists only when j and k are different frogs.
- recording_id contains the recording date and time (YYYY-MM-DD_HH-MM-SS); each recording is about one minute.
- Frog ids are assigned separately in each recording: frog 3 in one recording is not frog 3 in another.
  Always group by (recording_id, frog id), never by frog id alone.
- Times and durations are in milliseconds; video is 50 fps, so values are multiples of 20 ms.
- peak_call_hz_* is the peak frequency (Hz), peak_call_db_* the peak level (dB).
  spectral_diff_x_k = hz_k - hz_x and peak_call_db_diff_x_k = db_k - db_x.

Rules:
- Call get_schema before your first query.
- Never guess a number: every number in your answer must come from a tool result.
- Before answering, check that the data supports the question (sample sizes, units,
  whether values are comparable across recordings and dates). If it does not, say so clearly.
- When a figure helps, make one with run_python and save it to OUTPUT_DIR.
- End with a short answer: the finding, the numbers behind it, and any caveat."""

# Tool descriptions: this is all the model knows about your functions.
TOOLS = [
    {
        "name": "get_schema",
        "description": "Return the tables in the database and their columns. Call this before writing any SQL query.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "run_sql",
        "description": "Run one read-only SQLite query. Returns at most 50 rows, so aggregate "
                       "(COUNT, AVG, GROUP BY) instead of selecting raw rows.",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "A single SELECT statement."}},
            "required": ["query"],
        },
    },
    {
        "name": "run_python",
        "description": "Run Python for statistics or plots that SQL cannot do. Available: pd, np, plt, "
                       "query(sql) which returns a DataFrame, and OUTPUT_DIR. print() whatever you need to see. "
                       "Save figures with plt.savefig(f'{OUTPUT_DIR}/name.png', dpi=150, bbox_inches='tight').",
        "input_schema": {
            "type": "object",
            "properties": {"code": {"type": "string", "description": "Python code to execute."}},
            "required": ["code"],
        },
    },
]

# Map each tool name the model may request to the real Python function.
FUNCTIONS = {"get_schema": get_schema, "run_sql": run_sql, "run_python": run_python}


def run_tool(name, tool_input):
    """Run the tool the model asked for and always return text, even on errors."""
    try:
        return str(FUNCTIONS[name](**tool_input))[:6000]
    except Exception as error:
        return f"ERROR: {type(error).__name__}: {error}"


def run_agent(question, verbose=True):
    messages = [{"role": "user", "content": question}]  # the agent's memory: the whole conversation so far
    trace = []

    for step in range(1, MAX_STEPS + 1):
        # 1. Send the full conversation so far to the model
        response = client.messages.create(
            model=MODEL, max_tokens=4000, system=SYSTEM_PROMPT,
            tools=TOOLS, messages=messages,
        )
        messages.append({"role": "assistant", "content": response.content})

        # 2. If the model did not ask for a tool, this is the final answer
        if response.stop_reason != "tool_use":
            answer = "".join(b.text for b in response.content if b.type == "text")
            trace.append({"step": step, "answer": answer})
            break

        # 3. Run every tool it asked for and send the results back
        results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            output = run_tool(block.name, block.input)
            if verbose:
                print(f"\n[step {step}] {block.name}: {json.dumps(block.input, ensure_ascii=False)[:300]}")
                print(output[:400])
            trace.append({"step": step, "tool": block.name, "input": block.input, "output": output})
            # tool_use_id links each result to the request it answers
            results.append({"type": "tool_result", "tool_use_id": block.id, "content": output})
        messages.append({"role": "user", "content": results})
    else:
        answer = f"Stopped after {MAX_STEPS} steps without a final answer."
        trace.append({"step": MAX_STEPS, "answer": answer})

    # Save the full trace: it is the debugging tool and material for the slides
    os.makedirs("logs", exist_ok=True)
    with open(f"logs/run_{time.strftime('%Y%m%d_%H%M%S')}.json", "w", encoding="utf-8") as f:
        json.dump({"question": question, "steps": trace}, f, ensure_ascii=False, indent=2)
    return answer, trace


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "Which recording has the most transitions?"
    answer, trace = run_agent(question)
    print("\n=== ANSWER ===\n" + answer)