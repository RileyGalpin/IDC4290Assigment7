"""
agent.py - a tool-using agent on top of a local Ollama model.

Workflow:
  User Request -> LLM decides -> Tool Selection -> Tool Execution
               -> Observation -> (more tools, or) Final Response

Tools:
  1. run_sql      - query the dataset with read-only SQL (database tool)
  2. calculator   - exact arithmetic (numerical tool)
  3. make_chart   - draw a bar/line chart and save it as a PNG (visualization tool)

Usage:
  python agent.py "Which product had the highest total revenue?"
  python agent.py                      # interactive mode
  OLLAMA_MODEL=qwen2.5:7b python agent.py "..."   # try a different model
"""
import ast
import json
import operator
import os
import re
import sqlite3
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")  # render to files instead of opening a window
import matplotlib.pyplot as plt
import ollama
import pandas as pd

MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2")
DATA_PATH = os.environ.get("DATA_PATH", "sales.csv")
MAX_STEPS = 6  # safety limit so the agent can't loop forever
OPEN_CHARTS = True  # auto-open saved charts with the macOS `open` command

# ---------------------------------------------------------------------------
# Data setup: the CSV is loaded into pandas (for charts) and SQLite (for SQL)
# ---------------------------------------------------------------------------
df = pd.read_csv(DATA_PATH)
conn = sqlite3.connect(":memory:", check_same_thread=False)
df.to_sql("sales", conn, index=False, if_exists="replace")


def schema_text() -> str:
    cols = ", ".join(f"{c} ({t})" for c, t in zip(df.columns, df.dtypes.astype(str)))
    sample = df.head(3).to_string(index=False)
    return f"Table `sales` has {len(df)} rows. Columns: {cols}\nFirst rows:\n{sample}"


# ---------------------------------------------------------------------------
# Tool 1: SQL
# ---------------------------------------------------------------------------
def run_sql(query: str) -> str:
    q = query.strip().rstrip(";")
    if not re.match(r"(?is)^\s*(select|with)\b", q) or re.search(
        r"(?i)\b(insert|update|delete|drop|alter|create|attach|pragma)\b", q
    ):
        return "ERROR: only read-only SELECT queries are allowed."
    try:
        result = pd.read_sql_query(q, conn)
    except Exception as e:  # bad column names, syntax errors, etc.
        return f"ERROR: SQL failed: {e}"
    if result.empty:
        return "Query returned no rows."
    return result.head(50).to_string(index=False)


# ---------------------------------------------------------------------------
# Tool 2: calculator (safe arithmetic, no eval)
# ---------------------------------------------------------------------------
_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}


def _eval_node(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        return _BIN_OPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_eval_node(node.operand)
    raise ValueError("only numbers and + - * / ( ) are supported")


def calculator(expression: str) -> str:
    try:
        tree = ast.parse(expression.replace(",", "").replace("$", ""), mode="eval")
        return str(round(_eval_node(tree.body), 6))
    except ZeroDivisionError:
        return "ERROR: division by zero."
    except Exception as e:
        return f"ERROR: could not evaluate '{expression}': {e}"


# ---------------------------------------------------------------------------
# Tool 3: chart maker
# ---------------------------------------------------------------------------
_chart_count = 0


def make_chart(
    chart_type: str, x_column: str, y_column: str, title: str, group_by: str = None
) -> str:
    global _chart_count
    if chart_type not in ("bar", "line"):
        return "ERROR: chart_type must be 'bar' or 'line'."
    needed = [x_column, y_column] + ([group_by] if group_by else [])
    for col in needed:
        if col not in df.columns:
            return f"ERROR: unknown column '{col}'. Available columns: {list(df.columns)}"
    if not pd.api.types.is_numeric_dtype(df[y_column]):
        return f"ERROR: y_column '{y_column}' must be numeric."

    fig, ax = plt.subplots(figsize=(8, 5))
    if group_by:
        data = df.pivot_table(
            index=x_column, columns=group_by, values=y_column, aggfunc="sum"
        ).reindex(df[x_column].unique())  # keep original order (Jan, Feb, ...)
    else:
        data = df.groupby(x_column, sort=False)[y_column].sum()
    data.plot(kind=chart_type, ax=ax, marker="o" if chart_type == "line" else None)
    ax.set_title(title)
    ax.set_xlabel(x_column)
    ax.set_ylabel(f"total {y_column}")
    fig.tight_layout()

    _chart_count += 1
    filename = f"chart_{_chart_count}.png"
    fig.savefig(filename, dpi=150)
    plt.close(fig)
    if OPEN_CHARTS:
        try:
            subprocess.run(["open", filename], check=False)
        except FileNotFoundError:
            pass
    return f"Chart saved to {filename}."


# ---------------------------------------------------------------------------
# Tool registry + schemas (this is what the LLM sees when deciding what to call)
# ---------------------------------------------------------------------------
REGISTRY = {"run_sql": run_sql, "calculator": calculator, "make_chart": make_chart}

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "run_sql",
            "description": (
                "Run a read-only SQLite SELECT query on the `sales` table and return the "
                "result rows. Use this for ANY question about the data: totals, averages, "
                "maxima, rankings, comparisons, filters."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "A SQLite SELECT statement, e.g. SELECT product, SUM(units) FROM sales GROUP BY product",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": (
                "Evaluate an arithmetic expression exactly. Use this for percent change, "
                "ratios, or any math on numbers you already have."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Arithmetic expression, e.g. (150 - 120) / 120 * 100",
                    }
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "make_chart",
            "description": (
                "Create and save a chart from the sales data. Values in y_column are summed "
                "for each x_column value. Use when the user asks for a chart, plot, or graph."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "chart_type": {"type": "string", "enum": ["bar", "line"]},
                    "x_column": {"type": "string", "description": "Column for the x-axis, e.g. month or product"},
                    "y_column": {"type": "string", "description": "Numeric column to sum, e.g. revenue or units"},
                    "title": {"type": "string", "description": "Chart title"},
                    "group_by": {
                        "type": "string",
                        "description": "Optional column that splits the data into separate series, e.g. product",
                    },
                },
                "required": ["chart_type", "x_column", "y_column", "title"],
            },
        },
    },
]

SYSTEM_PROMPT = f"""You are a data analyst assistant with access to tools.

{schema_text()}

Rules:
- For any question about the data, call run_sql. Never guess or compute data values yourself.
- For arithmetic on numbers you already have (percent change, ratios), call calculator.
- If the user asks for a chart, plot, or graph, call make_chart.
- If a tool returns an ERROR, read the message, fix your call, and try again.
- If the question cannot be answered from the data (for example it asks WHY something happened, or needs a column that does not exist), say so instead of guessing.
- If no tool is needed (greetings, general knowledge), answer directly.
- Once you have what you need, give a short final answer that uses the exact numbers the tools returned."""


# ---------------------------------------------------------------------------
# The agent loop
# ---------------------------------------------------------------------------
def run_agent(user_request: str, verbose: bool = True) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_request},
    ]

    for step in range(1, MAX_STEPS + 1):
        # 1. LLM reasoning / decision: answer now, or call a tool?
        response = ollama.chat(
            model=MODEL, messages=messages, tools=TOOLS, options={"temperature": 0}
        )
        msg = response.message
        messages.append(msg)

        # No tool call -> the model is giving its final response
        if not msg.tool_calls:
            if verbose:
                print(f"[Step {step}] Final response")
            return msg.content

        # 2. Tool selection + execution + observation
        for call in msg.tool_calls:
            name = call.function.name
            args = dict(call.function.arguments)
            if verbose:
                print(f"[Step {step}] Action: {name}({json.dumps(args)})")

            func = REGISTRY.get(name)
            if func is None:
                result = f"ERROR: unknown tool '{name}'. Available: {list(REGISTRY)}"
            else:
                try:
                    result = func(**args)
                except TypeError as e:  # wrong/missing arguments
                    result = f"ERROR: bad arguments for {name}: {e}"

            if verbose:
                print(f"[Step {step}] Observation: {result}\n")
            messages.append({"role": "tool", "tool_name": name, "content": result})

    return "Stopped: reached the step limit without a final answer."


if __name__ == "__main__":
    print(f"Model: {MODEL} | Data: {DATA_PATH} ({len(df)} rows)\n")
    if len(sys.argv) > 1:
        print("\nAnswer:", run_agent(" ".join(sys.argv[1:])))
    else:
        print("Ask a question about the data (type 'quit' to exit).")
        while True:
            question = input("\nYou: ").strip()
            if question.lower() in ("quit", "exit", ""):
                break
            print("\nAnswer:", run_agent(question))
