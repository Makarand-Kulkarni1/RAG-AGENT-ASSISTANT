import json
from groq import Groq

from app.config import GROQ_API_KEY, CHAT_MODEL
from app.tools import TOOL_DEFINITIONS, AVAILABLE_TOOLS
from app.rag import retrieve

_client = Groq(api_key=GROQ_API_KEY)

SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions using the provided "
    "document context. You also have access to a calculator tool for any "
    "math the user asks about. Use the context and tools as needed; if "
    "something isn't in the context and doesn't need a calculation, say "
    "you don't know."
)


def run_agent(query: str, index, chunks: list, metadata: list) -> str:
    # Step 1: retrieve relevant context, same as Day 1
    results = retrieve(query, index, chunks, metadata)
    context = "\n\n".join(r["text"] for r in results)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
    ]

    # Step 2: first call — model decides whether to answer directly or call a tool
    response = _client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        tools=TOOL_DEFINITIONS,
        tool_choice="auto",
    )

    response_message = response.choices[0].message

    # Step 3: if the model didn't ask for a tool, just return its answer
    if not response_message.tool_calls:
        return response_message.content

    # Step 4: the model wants to call one or more tools — run them for real
    messages.append(response_message)

    for tool_call in response_message.tool_calls:
        func_name = tool_call.function.name
        func_args = json.loads(tool_call.function.arguments)

        if func_name in AVAILABLE_TOOLS:
            result = AVAILABLE_TOOLS[func_name](**func_args)
        else:
            result = f"Error: unknown tool {func_name}"

        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "name": func_name,
            "content": result,
        })

    # Step 5: send the tool result back so the model can give a final answer
    final_response = _client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
    )
    return final_response.choices[0].message.content