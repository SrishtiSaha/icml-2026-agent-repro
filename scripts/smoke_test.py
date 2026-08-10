from __future__ import annotations

import importlib.metadata as md
import importlib

from huggingface_hub import HfApi


def version(pkg: str) -> str:
    try:
        value = md.version(pkg)
    except md.PackageNotFoundError:
        return "not installed"
    if value:
        return value
    try:
        module = importlib.import_module(pkg.replace("-", "_"))
    except ImportError:
        return "installed (metadata version missing)"
    return str(getattr(module, "__version__", "installed (metadata version missing)"))


print("ICML 2026 repro environment smoke test")
print(f"trackio: {version('trackio')}")
print(f"huggingface_hub: {version('huggingface-hub')}")
print(f"openreview-py: {version('openreview-py')}")
print(f"arxiv: {version('arxiv')}")
print(f"datasets: {version('datasets')}")

try:
    user = HfApi().whoami()
except Exception as exc:
    print(f"hf auth: not logged in ({exc.__class__.__name__})")
else:
    print(f"hf auth: logged in as {user.get('name', '<unknown>')}")
