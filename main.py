"""Ask the agent one question from the command line.

Usage:  python main.py "Which recording has the most transitions?"
"""
import sys

from frog_agent.graph import run_agent

if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "Which recording has the most transitions?"
    answer, trace = run_agent(question)
    print("\n=== ANSWER ===\n" + answer)
