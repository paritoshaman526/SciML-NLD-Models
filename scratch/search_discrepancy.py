import json

log_path = r"C:\Users\satya\.gemini\antigravity-ide\brain\1c89c907-107d-4ec6-b918-cd19ff5817dc\.system_generated\logs\transcript.jsonl"

with open(log_path, "r", encoding="utf-8") as f:
    for idx, line in enumerate(f):
        data = json.loads(line)
        content = data.get("content", "")
        if "discrepancy" in content.lower():
            step = data.get("step_index", 0)
            src = data.get("source", "")
            print(f"\n--- STEP {step} ({src}) ---")
            c_low = content.lower()
            pos = 0
            while True:
                pos = c_low.find("discrepancy", pos)
                if pos == -1:
                    break
                start = max(0, pos - 150)
                end = min(len(content), pos + 150)
                snippet = content[start:end].replace("\n", " ").encode("ascii", "replace").decode("ascii")
                print(f"  [char {pos}]: ...{snippet}...")
                pos += len("discrepancy")
