"""Environment and runtime metadata capture and merge (Requirement 22)."""
from __future__ import annotations

import copy
import json
import os
import platform
import subprocess
import sys
import tempfile
from collections import Counter
from contextlib import contextmanager
from datetime import datetime
from importlib import metadata
from pathlib import Path
from typing import Iterator

from common.config import OFFICIAL_DIR, resolve_limit, results_root

KEY_PACKAGES: tuple[str, ...] = (
    "openai", "pandas", "scipy", "statsmodels", "nltk", "sqlparse",
)
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
ENV_FILENAME = "environment.json"


def _read(path: str) -> str:
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _is_wsl() -> bool:
    return (
        "microsoft" in platform.release().lower()
        or "WSL_DISTRO_NAME" in os.environ
        or "microsoft" in _read("/proc/version").lower()
    )


def _distro() -> str | None:
    for line in _read("/etc/os-release").splitlines():
        if line.startswith("PRETTY_NAME="):
            return line.split("=", 1)[1].strip().strip('"')
    return os.environ.get("WSL_DISTRO_NAME")


def _cpu_model() -> str | None:
    for line in _read("/proc/cpuinfo").splitlines():
        if line.lower().startswith("model name"):
            return line.split(":", 1)[1].strip()
    return platform.processor() or None


def _ram_gib() -> float | None:
    for line in _read("/proc/meminfo").splitlines():
        if line.startswith("MemTotal:"):
            kib = int(line.split()[1])
            return round(kib / (1024 ** 2), 2)
    return None


def collect_machine() -> dict:
    """OS, kernel, WSL flag, distro, CPU, logical cores, RAM (GiB), Python (Req 22.1-22.3)."""
    is_wsl = _is_wsl()
    return {
        "os": platform.system(),
        "kernel": platform.release(),
        "is_wsl": is_wsl,
        "distro": _distro(),
        "cpu_model": _cpu_model(),
        "logical_cpus": os.cpu_count(),
        "ram_gib": _ram_gib(),
        "ram_note": (
            "RAM allocated to the WSL VM (from /proc/meminfo), not total host RAM"
            if is_wsl else "Total RAM from /proc/meminfo"
        ),
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
    }


def collect_packages(names=KEY_PACKAGES) -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for n in names:
        try:
            out[n] = metadata.version(n)
        except metadata.PackageNotFoundError:
            out[n] = None
    return out


def official_commit() -> str | None:
    try:
        r = subprocess.run(
            ["git", "-C", str(OFFICIAL_DIR), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    sha = r.stdout.strip()
    return sha if r.returncode == 0 and sha else None


def inference_info() -> dict:
    return {
        "access": "api",
        "service": "OpenRouter",
        "base_url": OPENROUTER_BASE_URL,
        "remote_hardware_controlled": False,
        "local_role": "orchestration, SQL execution, evaluation",
    }


# ---------------------------------------------------------------- persistence

def load_env(path: Path) -> dict:
    path = Path(path)
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_env(path: Path, data: dict) -> None:
    """Atomic write: temp file in the same directory, then os.replace."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".env_", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, sort_keys=True, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def merge_env(old: dict, new: dict) -> dict:
    """Pure merge. Snapshots replaced; runs unioned by (name, started_at); providers deep-merged."""
    out = copy.deepcopy(old) if old else {}
    new = copy.deepcopy(new) if new else {}

    for key in ("machine", "packages", "inference"):
        if key in new:
            out[key] = new[key]
    if "official_commit" in new:
        out["official_commit"] = new["official_commit"]

    runs: dict[tuple, dict] = {}
    for r in list(out.get("runs", [])) + list(new.get("runs", [])):
        runs[(r.get("name"), r.get("started_at"))] = r
    if runs or "runs" in out or "runs" in new:
        out["runs"] = sorted(
            runs.values(), key=lambda r: (str(r.get("started_at")), str(r.get("name")))
        )

    if "models" in new or "models" in out:
        models = out.setdefault("models", {})
        for mid, info in new.get("models", {}).items():
            cur = models.setdefault(mid, {})
            for k, v in info.items():
                if k == "providers":
                    provs = cur.setdefault("providers", {})
                    for pname, pinfo in v.items():
                        merged = dict(provs.get(pname, {}))
                        merged.update(pinfo)
                        provs[pname] = merged
                else:
                    cur[k] = v

    for k, v in new.items():
        if k not in ("machine", "packages", "inference", "official_commit", "runs", "models"):
            out[k] = v
    return out


def env_path(limit: int | None = None) -> Path:
    return results_root(limit) / ENV_FILENAME


def snapshot() -> dict:
    return {
        "machine": collect_machine(),
        "packages": collect_packages(),
        "official_commit": official_commit(),
        "inference": inference_info(),
    }


def _update(path: Path, new: dict) -> dict:
    merged = merge_env(load_env(path), new)
    save_env(path, merged)
    return merged


# ---------------------------------------------------------------- recording

def record_models(
    model_id: str,
    providers: Counter,
    endpoints: list[dict],
    limit: int | None = None,
    path: Path | None = None,
) -> None:
    """Record serving providers (with call counts) and their reported quantization (Req 22.6)."""
    quant: dict[str, str | None] = {}
    for ep in endpoints or []:
        name = ep.get("provider_name") or ep.get("provider") or ep.get("name")
        if name:
            quant[str(name)] = ep.get("quantization")
    provs = {
        str(p): {"calls": int(n), "quantization": quant.get(str(p))}
        for p, n in dict(providers).items()
    }
    p = Path(path) if path is not None else env_path(resolve_limit(limit))
    _update(p, {**snapshot(), "models": {model_id: {"providers": provs}}})


def _now() -> str:
    return datetime.now().astimezone().isoformat()


@contextmanager
def track_run(
    name: str,
    limit: int | None,
    extra: dict | None = None,
    path: Path | None = None,
) -> Iterator[None]:
    """Record a runs[] entry on both normal exit and exception (Req 22.4)."""
    p = Path(path) if path is not None else env_path(limit)
    entry = {"name": name, "limit": limit, "started_at": _now()}
    if extra:
        entry["extra"] = extra
    status = "failed"
    try:
        yield
        status = "completed"
    finally:
        entry["status"] = status
        entry["finished_at"] = _now()
        if status == "failed":
            entry["error"] = repr(sys.exc_info()[1]) if sys.exc_info()[1] else None
        _update(p, {**snapshot(), "runs": [entry]})
