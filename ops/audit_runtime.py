"""Read-only runtime audit; no usernames, network peers or environment variables."""
import importlib.util
import json
import shutil
import subprocess


def command(args):
    try:
        result = subprocess.run(args, capture_output=True, timeout=30)
        raw = result.stdout + result.stderr
        encoding = "utf-16-le" if b"\x00" in raw else "utf-8"
        return {"exit_code": result.returncode, "text": raw.decode(encoding, errors="replace").strip()}
    except (OSError, subprocess.TimeoutExpired):
        return {"error": "unavailable_or_timeout"}


output = {"ssh": command(["ssh", "-V"]) if shutil.which("ssh") else None,
          "wsl": {"status": command(["wsl", "--status"]), "distributions": command(["wsl", "--list", "--quiet"])} if shutil.which("wsl") else None}
try:
    exists = importlib.util.find_spec("torch") is not None
    output["pytorch"] = {"installed": exists}
    if exists:
        import torch
        output["pytorch"].update(version=torch.__version__, built_cuda=torch.version.cuda,
                                 cuda_available=torch.cuda.is_available(),
                                 devices=[{"name": torch.cuda.get_device_properties(i).name,
                                           "vram_bytes": torch.cuda.get_device_properties(i).total_memory}
                                          for i in range(torch.cuda.device_count())])
except Exception:
    output["pytorch"] = {"error": "import_or_gpu_probe_failed"}
print(json.dumps(output))
