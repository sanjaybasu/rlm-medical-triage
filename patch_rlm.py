"""
Patches rlm/clients/openai.py to add max_tokens=2048 to all
chat.completions.create() calls. Run during Modal image build to prevent
DeepSeek-R1-70B from generating unbounded thinking chains in REPL iterations
(which can exceed 10K tokens per call and stall the pipeline for hours).
"""
import site, os, re

for pkg_dir in site.getsitepackages():
    path = os.path.join(pkg_dir, "rlm", "clients", "openai.py")
    if os.path.exists(path):
        break
else:
    raise RuntimeError("Could not locate rlm/clients/openai.py")

src = open(path).read()
patched = src.replace(
    "model=model, messages=messages, extra_body=extra_body)",
    "model=model, messages=messages, extra_body=extra_body, max_tokens=2048)",
)
if patched == src:
    print("WARNING: pattern not found in", path, "— no patch applied")
else:
    open(path, "w").write(patched)
    print("Patched:", path, "(max_tokens=2048 added to all create() calls)")
