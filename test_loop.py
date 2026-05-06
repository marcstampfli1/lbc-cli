"""Drive the agent's ReAct loop against local Ollama, no stdin."""
import os
os.environ["OPENAI_BASE_URL"] = "http://localhost:11434/v1"
os.environ["OPENAI_API_KEY"] = "ollama"
os.environ["AGENT_MODEL"] = "qwen2.5:3b"
os.environ["LIBRECHAT_URL"] = ""

import agent

backend = agent.OpenAIBackend()
messages = [
    {"role": "system", "content": agent.SYSTEM_PROMPT},
    {"role": "user", "content": "Use list_dir to show what's in /tmp, then summarize in one sentence."},
]

for i in range(8):
    print(f"\n--- turn {i} ---")
    text = backend.send(messages)
    print(f"[assistant]\n{text}\n")
    messages.append({"role": "assistant", "content": text})
    calls = agent.extract_tool_calls(text)
    if not calls:
        print("(no tool calls — loop ends)")
        break
    results = []
    for name, args in calls:
        print(f"[tool] {name}({args})")
        result = agent.call_tool(name, args)
        print(f"[result]\n{result[:400]}\n")
        results.append(f"<tool_response>\n{result}\n</tool_response>")
    messages.append({"role": "user", "content": "\n".join(results)})
