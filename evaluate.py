"""Run every evaluation question through the agent, grade the answers and save the results.

Usage:  python evaluate.py
Outputs: outputs/eval_results.csv and outputs/eval_accuracy.png
"""
import os
import re
import time

import anthropic
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from eval_questions import QUESTIONS
from frog_agent.config import MODEL, OUTPUT_DIR
from frog_agent.graph import run_agent

client = anthropic.Anthropic()  # used only by the judge; the API key comes from .env, loaded by frog_agent.graph


def numbers_in(text):
    """Extract every number from the answer text (commas removed: 12,910 -> 12910)."""
    return [float(n) for n in re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))]


def judge(question, answer, rubric):
    """Ask the model to grade an answer against a rubric. Returns (passed, reason)."""
    prompt = (f"Question: {question}\n\nAnswer to grade:\n{answer}\n\nGrading rubric: {rubric}\n\n"
              "Reply with PASS or FAIL on the first line, then one sentence explaining why.")
    response = client.messages.create(model=MODEL, max_tokens=200,
                                      messages=[{"role": "user", "content": prompt}])
    verdict = response.content[0].text.strip()
    return verdict.upper().startswith("PASS"), verdict


def grade(item, answer):
    """Return (passed, note) for one question."""
    if item["type"] == "number":
        found = numbers_in(answer)
        passed = any(abs(x - item["expected"]) <= item["tolerance"] for x in found)
        return passed, f"expected {item['expected']}"
    if item["type"] == "contains":
        return item["expected"] in answer, f"expected '{item['expected']}'"
    return judge(item["question"], answer, item["rubric"])


rows = []
for item in QUESTIONS:
    print(f"\n=== {item['id']} [{item['category']}] {item['question']}")
    start = time.time()
    answer, trace = run_agent(item["question"], verbose=False)
    passed, note = grade(item, answer)
    tool_calls = [s for s in trace if "tool" in s]
    errors = sum("ERROR:" in str(s.get("output", "")) for s in tool_calls)
    rows.append({
        "id": item["id"],
        "category": item["category"],
        "question": item["question"],
        "passed": passed,
        "note": note,
        "tool_calls": len(tool_calls),
        "tool_errors": errors,
        # share of tool calls that worked on the first try (no error came back)
        "first_try_ok": round(1 - errors / len(tool_calls), 2) if tool_calls else 1.0,
        "tokens": sum(s["tokens"] for s in trace if "tokens" in s),
        # times the verify node sent the answer back because a number had no source
        "verify_rejections": sum(s.get("verify") == "rejected" for s in trace),
        "seconds": round(time.time() - start, 1),
        "answer": answer,
    })
    print("PASS" if passed else "FAIL", "|", note)

# Save the table and print the summary
os.makedirs(OUTPUT_DIR, exist_ok=True)
results = pd.DataFrame(rows)
results.to_csv(OUTPUT_DIR / "eval_results.csv", index=False)
summary = results.groupby("category")["passed"].agg(["mean", "count"])
print("\n" + summary.to_string())
print(f"\nOverall accuracy: {results['passed'].mean():.0%} ({results['passed'].sum()}/{len(results)})")
print(f"Mean tool calls per question: {results['tool_calls'].mean():.1f}")
print(f"Tool calls that worked on the first try: {1 - results['tool_errors'].sum() / results['tool_calls'].sum():.0%}")
print(f"Mean tokens per question: {results['tokens'].mean():,.0f}")
print(f"Verify rejections: {results['verify_rejections'].sum()} in {(results['verify_rejections'] > 0).sum()} questions")

# One bar chart for the slides: accuracy by question category
order = ["lookup", "aggregation", "data_quality", "trap"]
summary = summary.reindex([c for c in order if c in summary.index])
fig, ax = plt.subplots(figsize=(6, 3.5))
bars = ax.bar(summary.index, summary["mean"] * 100, color="#4C72B0")
for bar, (_, r) in zip(bars, summary.iterrows()):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
            f"{r['mean']:.0%}\n(n={int(r['count'])})", ha="center", fontsize=9)
ax.set_ylim(0, 115)
ax.set_ylabel("Accuracy (%)")
ax.set_title("Agent accuracy by question type")
ax.spines[["top", "right"]].set_visible(False)
plt.savefig(OUTPUT_DIR / "eval_accuracy.png", dpi=150, bbox_inches="tight")
print("Saved outputs/eval_results.csv and outputs/eval_accuracy.png")