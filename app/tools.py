def calculator(expression: str) -> str:
    """
    Safely evaluates a basic arithmetic expression.
    Only used by the agent when it decides math is needed.
    """
    try:
        allowed_chars = set("0123456789+-*/(). ")
        if not all(c in allowed_chars for c in expression):
            return "Error: expression contains disallowed characters"
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"Error: {e}"


TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate a basic arithmetic expression, e.g. '15 * 3' or '(20-5)/3'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "The arithmetic expression to evaluate",
                    }
                },
                "required": ["expression"],
            },
        },
    }
]

AVAILABLE_TOOLS = {
    "calculator": calculator,
}