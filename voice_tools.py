"""VaaniAI tools: local function-calling tools the voice agent can use."""

import ast
import json
import operator
from datetime import datetime
from pathlib import Path

NOTES_FILE = Path(__file__).resolve().parent / "notes.json"

# ---------------------------------------------------------------------- #
# Tool implementations
# ---------------------------------------------------------------------- #

_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub,
    ast.Mult: operator.mul, ast.Div: operator.truediv,
    ast.Pow: operator.pow, ast.Mod: operator.mod,
    ast.USub: operator.neg, ast.UAdd: operator.pos,
    ast.FloorDiv: operator.floordiv,
}


def _safe_eval(node):
    """Evaluate a pure-arithmetic AST. No names, no calls, no attributes."""
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        left = _safe_eval(node.left)
        right = _safe_eval(node.right)
        return _OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError("only plain arithmetic is allowed")


def calculate(expression: str) -> str:
    """Safely evaluate a plain arithmetic expression like (12+8)*3/2."""
    expression = (expression or "").strip()
    if not expression or len(expression) > 200:
        return "Error: give me a short arithmetic expression."
    try:
        tree = ast.parse(expression, mode="eval")
        result = _safe_eval(tree)
    except ZeroDivisionError:
        return "Error: division by zero."
    except (ValueError, SyntaxError):
        return ("Error: only plain arithmetic (+, -, *, /, %, **) "
                "is supported.")
    value = float(result)
    if value.is_integer():
        return f"{expression} = {int(value)}"
    return f"{expression} = {round(value, 6)}"


def get_current_time() -> str:
    now = datetime.now()
    return now.strftime("It's %I:%M %p on %A, %d %B %Y.").lstrip("0")


def _load_notes():
    if NOTES_FILE.exists():
        try:
            return json.loads(NOTES_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return []
    return []


def add_note(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return "Error: the note text must not be empty."
    notes = _load_notes()
    notes.append({"id": (notes[-1]["id"] + 1) if notes else 1,
                  "text": text,
                  "created": datetime.now().isoformat(timespec="seconds")})
    NOTES_FILE.write_text(json.dumps(notes, indent=2), encoding="utf-8")
    return f"Noted: {text}"


def list_notes() -> str:
    notes = _load_notes()
    if not notes:
        return "No notes yet."
    return "\n".join(f"#{n['id']} {n['text']}" for n in notes)


def delete_note(note_id: int) -> str:
    notes = _load_notes()
    remaining = [n for n in notes if n["id"] != note_id]
    if len(remaining) == len(notes):
        return f"Error: no note with id {note_id}."
    NOTES_FILE.write_text(json.dumps(remaining, indent=2), encoding="utf-8")
    return f"Deleted note #{note_id}"


# ---------------------------------------------------------------------- #
# Tool registry (OpenAI function-calling schema + executor)
# ---------------------------------------------------------------------- #

TOOLS = {
    "get_current_time": {
        "impl": get_current_time,
        "schema": {
            "type": "function",
            "function": {
                "name": "get_current_time",
                "description": "Get the current local date and time.",
                "parameters": {"type": "object", "properties": {}},
            },
        },
    },
    "calculate": {
        "impl": calculate,
        "schema": {
            "type": "function",
            "function": {
                "name": "calculate",
                "description": "Evaluate a plain arithmetic expression "
                               "like (12+8)*3/2. Use for any math.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "The arithmetic expression",
                        },
                    },
                    "required": ["expression"],
                },
            },
        },
    },
    "add_note": {
        "impl": add_note,
        "schema": {
            "type": "function",
            "function": {
                "name": "add_note",
                "description": "Save a short note for the user "
                               "(reminders, ideas, to-dos).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string",
                                 "description": "The note to save"},
                    },
                    "required": ["text"],
                },
            },
        },
    },
    "list_notes": {
        "impl": list_notes,
        "schema": {
            "type": "function",
            "function": {
                "name": "list_notes",
                "description": "List all saved notes.",
                "parameters": {"type": "object", "properties": {}},
            },
        },
    },
    "delete_note": {
        "impl": delete_note,
        "schema": {
            "type": "function",
            "function": {
                "name": "delete_note",
                "description": "Delete a note by its id (from list_notes).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "note_id": {"type": "integer",
                                    "description": "The note id"},
                    },
                    "required": ["note_id"],
                },
            },
        },
    },
}


def schemas():
    return [t["schema"] for t in TOOLS.values()]


def execute(name, arguments):
    """Run a tool by name. Returns (ok, result_text).

    ok is False for unknown tools, bad arguments, and tools that return an
    "Error: ..." string — so the UI can show failures distinctly.
    """
    tool = TOOLS.get(name)
    if tool is None:
        return False, f"Error: unknown tool {name}"
    try:
        result = str(tool["impl"](**arguments))
    except TypeError as e:
        return False, f"Error: bad arguments for {name}: {e}"
    except Exception as e:                       # noqa: BLE001
        return False, f"Error: {name} failed: {e}"
    if result.startswith("Error:"):
        return False, result
    return True, result
