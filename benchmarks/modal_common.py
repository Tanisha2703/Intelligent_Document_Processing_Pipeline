"""Cost guard and spend ledger shared by the Modal scripts (no modal import: usable and testable anywhere)."""

import json
import os
import pathlib
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
LEDGER = ROOT / "results" / "raw" / "modal_spend.json"
BUDGET = float(os.environ.get("DOCINTEL_BUDGET_USD", "15"))

# Modal list prices in USD per second (modal.com/pricing). Check again before relying on them.
GPU_PRICE = {
    "T4": 0.000164,
    "L4": 0.000222,
    "A10G": 0.000306,
    "A10": 0.000306,
    "L40S": 0.000542,
    "A100-40GB": 0.000583,
    "A100-80GB": 0.000694,
    "H100": 0.001097,
}
CPU_PRICE, MEM_PRICE = 0.0000131, 0.00000222  # per core-second, per GiB-second


def cost_per_second(gpu: str, cpu_cores: int = 4, mem_mib: int = 16384) -> float:
    return GPU_PRICE[gpu] + cpu_cores * CPU_PRICE + (mem_mib / 1024) * MEM_PRICE


def spent() -> float:
    return sum(e["est_cost_usd"] for e in json.loads(LEDGER.read_text())) if LEDGER.exists() else 0.0


def ledger(entry: dict) -> None:
    rows = json.loads(LEDGER.read_text()) if LEDGER.exists() else []
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    LEDGER.write_text(json.dumps(rows + [{"when": time.strftime("%Y-%m-%d %H:%M:%S"), **entry}], indent=1))


def guard(label: str, worst_seconds: float, gpu: str, cpu_cores: int = 4, mem_mib: int = 16384) -> None:
    """Refuse (SystemExit) unless the WORST-CASE cost fits in what is left of the budget after the ledger."""
    worst = worst_seconds * cost_per_second(gpu, cpu_cores, mem_mib)
    left = BUDGET - spent()
    print(
        f"{label}: worst case ${worst:.2f} on {gpu} ({worst_seconds / 60:.0f} min); budget ${BUDGET:.2f}, "
        f"spent ${spent():.2f}, left ${left:.2f}"
    )
    if worst > left:
        raise SystemExit(
            "worst-case cost exceeds the remaining budget (DOCINTEL_BUDGET_USD); aborting before launching anything"
        )
