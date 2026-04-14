from pathlib import Path

p = Path("/workspace/.cache/huggingface/hub/models--google--gemma-4-31B-it/refs")
p.mkdir(parents=True, exist_ok=True)
(p / "main").write_text("test")
print("OK:", (p / "main").read_text())
