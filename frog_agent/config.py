from pathlib import Path

# Project root = the folder above frog_agent/, so paths work from anywhere
ROOT = Path(__file__).resolve().parent.parent

DB_PATH = ROOT / "data" / "frogs.db"
OUTPUT_DIR = ROOT / "outputs"
LOG_DIR = ROOT / "logs"

MODEL = "claude-sonnet-5-5"
MAX_TOKENS = 4000

MAX_STEPS = 15            # becomes LangGraph's recursion_limit
MAX_ROWS = 50             # rows returned by run_sql
MAX_TOOL_OUTPUT = 6000    # characters of tool output sent back to the model
PYTHON_TIMEOUT_S = 30     # run_python runs in its own process and is stopped after this
MAX_VERIFY_RETRIES = 2    # times verify may send an answer back before giving up
