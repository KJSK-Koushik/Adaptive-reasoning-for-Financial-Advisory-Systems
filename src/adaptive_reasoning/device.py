"""Device detection and hardware reporting.

Phase 3 is the only GPU-hungry step. This module lets every other phase run
unchanged on CPU, and tells the user honestly what they are working with.
"""

from __future__ import annotations

import platform
import shutil
import subprocess
from dataclasses import dataclass, field


@dataclass
class HardwareInfo:
    device: str                      # resolved torch device string
    backend: str                     # cuda | xpu | mps | cpu
    gpu_name: str | None = None
    gpu_memory_gb: float | None = None
    cpu: str = ""
    cpu_threads: int = 0
    ram_gb: float = 0.0
    torch_version: str | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def can_generate_traces_locally(self) -> bool:
        """Phase 3 needs real GPU throughput to be practical at full scale."""
        return self.backend == "cuda" and (self.gpu_memory_gb or 0) >= 6.0


def _nvidia_smi() -> tuple[str, float] | None:
    if not shutil.which("nvidia-smi"):
        return None
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=15,
            check=True,
        ).stdout.strip()
    except (subprocess.SubprocessError, OSError):
        return None
    if not out:
        return None
    name, mem = out.splitlines()[0].split(",")
    return name.strip(), float(mem) / 1024.0


def detect(preference: str = "auto") -> HardwareInfo:
    """Resolve the compute device, honouring an explicit preference when given."""
    info = HardwareInfo(device="cpu", backend="cpu")
    info.cpu = platform.processor() or platform.machine()

    try:
        import torch
    except ImportError:
        info.notes.append("torch is not installed; run `pip install -r requirements.txt`")
        return info

    info.torch_version = torch.__version__
    info.cpu_threads = torch.get_num_threads()

    available: list[str] = []
    if torch.cuda.is_available():
        available.append("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        available.append("mps")
    if hasattr(torch, "xpu") and torch.xpu.is_available():  # Intel Arc / Iris via IPEX
        available.append("xpu")

    if preference != "auto":
        if preference == "cpu":
            chosen = "cpu"
        elif preference in available:
            chosen = preference
        else:
            info.notes.append(
                f"requested device {preference!r} is unavailable; falling back to cpu"
            )
            chosen = "cpu"
    else:
        chosen = available[0] if available else "cpu"

    info.backend = chosen
    info.device = chosen

    if chosen == "cuda":
        props = torch.cuda.get_device_properties(0)
        info.gpu_name = props.name
        info.gpu_memory_gb = round(props.total_memory / 1024**3, 2)
    elif chosen == "cpu":
        smi = _nvidia_smi()
        if smi:
            info.notes.append(
                f"an NVIDIA GPU ({smi[0]}) was detected but torch has no CUDA support - "
                "you likely installed the CPU build of torch"
            )
        else:
            info.notes.append(
                "no CUDA GPU found; Phase 3 trace generation should run on Kaggle, "
                "Colab, or a rented GPU"
            )

    try:
        import psutil  # optional

        info.ram_gb = round(psutil.virtual_memory().total / 1024**3, 1)
    except ImportError:
        pass

    return info


def torch_dtype(name: str, backend: str):
    """Map a config dtype string to a torch dtype, downgrading if unsupported on CPU."""
    import torch

    mapping = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    dtype = mapping[name]
    if backend == "cpu" and dtype is torch.float16:
        # float16 matmul is not properly supported on most CPUs.
        return torch.float32
    return dtype
