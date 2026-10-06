import json

log_path = r"C:\Users\satya\.gemini\antigravity-ide\brain\1c89c907-107d-4ec6-b918-cd19ff5817dc\.system_generated\logs\transcript.jsonl"

with open(log_path, "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)
        if data.get("step_index") == 1854:
            content = data.get("content", "").encode("ascii", "replace").decode("ascii")
            print(content)
