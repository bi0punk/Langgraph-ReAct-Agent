# src/agent/tools/system_stats.py
# ------------------------------------------------------------
# Tool: system_stats
# - Lee uso de CPU y memoria RAM actuales (vía psutil)
# - Devuelve un resumen de texto
# - Registra cada invocación en un CSV (append)
#   Ruta: METRICS_CSV o ./data/metrics/system_stats.csv
# ------------------------------------------------------------

from __future__ import annotations
import os
import csv
from datetime import datetime
from pathlib import Path
from typing import Optional

from langchain_core.tools import tool

try:
    import psutil  # type: ignore
except Exception as e:
    raise RuntimeError(
        "La tool system_stats requiere 'psutil'. Instálalo con: pip install psutil"
    ) from e


def _csv_path() -> Path:
    path = os.getenv("METRICS_CSV", "./data/metrics/system_stats.csv")
    p = Path(path).expanduser().resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _append_csv(ts: str, cpu: float, mem_used_mb: float, mem_total_mb: float, note: str) -> None:
    csv_path = _csv_path()
    file_exists = csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if not file_exists:
            w.writerow(["timestamp", "cpu_percent", "mem_used_mb", "mem_total_mb", "note"])
        w.writerow([ts, f"{cpu:.2f}", f"{mem_used_mb:.2f}", f"{mem_total_mb:.2f}", note])


@tool("system_stats")
def system_stats(note: Optional[str] = "") -> str:
    """
    Retorna el uso actual de CPU y RAM, y guarda un registro en CSV.
    Argumentos:
      - note (opcional): etiqueta o comentario para el registro.
    Respuesta: texto legible con CPU% y memoria usada/total en MB.
    """
    ts = datetime.now().isoformat(timespec="seconds")

    cpu = psutil.cpu_percent(interval=0.2)
    vm = psutil.virtual_memory()
    mem_used_mb = vm.used / (1024 * 1024)
    mem_total_mb = vm.total / (1024 * 1024)

    _append_csv(ts, cpu, mem_used_mb, mem_total_mb, note or "")

    return (
        f"[{ts}] CPU: {cpu:.2f}% | RAM: {mem_used_mb:.2f} MB / {mem_total_mb:.2f} MB"
        + (f" | nota: {note}" if note else "")
    )
