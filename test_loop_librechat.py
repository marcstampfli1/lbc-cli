"""End-to-end: drive the agent's ReAct loop through LibreChat → Ollama → Qwen."""
import agent

backend = agent.LibreChatBackend()
print(f"connected to {agent.LIBRECHAT_URL}, endpoint={agent.LIBRECHAT_ENDPOINT}, model={agent.LIBRECHAT_MODEL}")

messages = [
    {"role": "system", "content": agent.SYSTEM_PROMPT},
    {"role": "user", "content": "Use list_dir on /etc and tell me how many entries there are."},
]

for i in range(8):
    print(f"\n--- turn {i} ---")
    text = backend.send(messages)
    print(f"[assistant]\n{text!r}\n")
    messages.append({"role": "assistant", "content": text})
    calls = agent.extract_tool_calls(text)
    if not calls:
        print("(no tool calls — loop ends)")
        break
    results = []
    for name, args in calls:
        print(f"[tool] {name}({args})")
        result = agent.call_tool(name, args)
        print(f"[result]\n{result[:300]}\n")
        results.append(f"<tool_response>\n{result}\n</tool_response>")
    messages.append({"role": "user", "content": "\n".join(results)})
