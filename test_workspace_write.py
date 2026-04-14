from pathlib import Path

# Test 1: write directly to /workspace root
p1 = Path("/workspace/direct_test.txt")
p1.write_text("hello")
print("Test 1 (direct):", p1.read_text())
p1.unlink()

# Test 2: write to a fresh subdirectory
p2 = Path("/workspace/hf_download/refs")
p2.mkdir(parents=True, exist_ok=True)
(p2 / "main").write_text("test")
print("Test 2 (subdir):", (p2 / "main").read_text())

# Test 3: write to the .cache path HF uses
p3 = Path("/workspace/.cache/huggingface/hub/models--google--gemma-4-31B-it/refs")
p3.mkdir(parents=True, exist_ok=True)
(p3 / "main").write_text("test")
print("Test 3 (.cache):", (p3 / "main").read_text())
