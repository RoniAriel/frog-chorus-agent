"""The agent's tools. Each tool's name, description and Pydantic schema are all the model knows about it."""
import json
import os
import sqlite3
import subprocess
import sys

import pandas as pd
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from frog_agent.config import DB_PATH, MAX_ROWS, MAX_TOOL_OUTPUT, OUTPUT_DIR, PYTHON_TIMEOUT_S


def _connect():
    # Read-only connection: the agent can never change the data
    return sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)


class SqlInput(BaseModel):
    query: str = Field(description="A single SELECT statement.")


class PythonInput(BaseModel):
    code: str = Field(description="Python code to execute.")


@tool
def get_schema() -> str:
    """Return the tables in the database and their columns. Call this before writing any SQL query."""
    con = _connect()
    lines = []
    tables = con.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    for (table,) in tables:
        n_rows = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        columns = con.execute(f"PRAGMA table_info({table})").fetchall()
        col_text = ", ".join(f"{c[1]} ({c[2]})" for c in columns)
        lines.append(f"TABLE {table} ({n_rows} rows): {col_text}")
    con.close()
    return "\n".join(lines)[:MAX_TOOL_OUTPUT]


@tool(args_schema=SqlInput)
def run_sql(query: str) -> str:
    """Run one read-only SQLite query. Returns at most 50 rows, so aggregate
    (COUNT, AVG, GROUP BY) instead of selecting raw rows."""
    try:
        con = _connect()
        df = pd.read_sql_query(query, con)
        con.close()
    except Exception as error:
        return f"ERROR: {error}"
    return df.head(MAX_ROWS).to_string(index=False)[:MAX_TOOL_OUTPUT]


# The code the model writes runs in a separate Python process with a timeout.
# This is an isolated process, not a sandbox: the process can still read and write files.
_RUNNER = """
import contextlib, io, json, sqlite3, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

settings = json.loads(sys.argv[1])

def query(sql):
    con = sqlite3.connect(f"file:{settings['db_path']}?mode=ro", uri=True)
    try:
        return pd.read_sql_query(sql, con)
    finally:
        con.close()

namespace = {"pd": pd, "np": np, "plt": plt, "query": query, "OUTPUT_DIR": settings["output_dir"]}
exec(sys.stdin.read(), namespace)
"""


@tool(args_schema=PythonInput)
def run_python(code: str) -> str:
    """Run Python for statistics or plots that SQL cannot do. Available: pd, np, plt,
    query(sql) which returns a DataFrame, and OUTPUT_DIR. print() whatever you need to see.
    Save figures with plt.savefig(f'{OUTPUT_DIR}/name.png', dpi=150, bbox_inches='tight')."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    settings = json.dumps({"db_path": str(DB_PATH), "output_dir": str(OUTPUT_DIR)})
    try:
        # -I: isolated mode, ignores PYTHON* environment variables and the user's site-packages
        result = subprocess.run([sys.executable, "-I", "-c", _RUNNER, settings], input=code,
                                capture_output=True, text=True, timeout=PYTHON_TIMEOUT_S, cwd=OUTPUT_DIR)
    except subprocess.TimeoutExpired:
        return f"ERROR: TimeoutExpired: the code ran longer than {PYTHON_TIMEOUT_S} seconds and was stopped."
    output = result.stdout
    if result.returncode != 0:
        # Keep only the last line of the traceback, e.g. "ZeroDivisionError: division by zero"
        last_line = result.stderr.strip().splitlines()[-1] if result.stderr.strip() else "unknown error"
        output += f"\nERROR: {last_line}"
    return (output or "(code ran, nothing was printed)")[:MAX_TOOL_OUTPUT]


TOOLS = [get_schema, run_sql, run_python]
for _tool in TOOLS:
    _tool.description = " ".join(_tool.description.split())  # docstring line breaks -> one clean line


if __name__ == "__main__":
    print(get_schema.invoke({}))
    print()
    print(run_sql.invoke({"query": """
        SELECT recording_id, COUNT(*) AS n_rows
        FROM transitions
        GROUP BY recording_id
        ORDER BY n_rows DESC
    """}))
