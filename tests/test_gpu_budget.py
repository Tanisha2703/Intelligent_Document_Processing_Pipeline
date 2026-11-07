"""The cost guard that stops a GPU job from starting when its worst case would exceed the budget."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

from src.utils import gpu_budget  # noqa: E402


def test_cost_per_second_adds_gpu_cpu_and_memory():
    expected = gpu_budget.GPU_PRICE["A10G"] + 4 * gpu_budget.CPU_PRICE + 16 * gpu_budget.MEM_PRICE
    assert gpu_budget.cost_per_second("A10G", 4, 16384) == pytest.approx(expected)


def test_guard_allows_a_job_that_fits_and_refuses_one_that_does_not(tmp_path, monkeypatch):
    monkeypatch.setattr(gpu_budget, "LEDGER", tmp_path / "spend.json")
    monkeypatch.setattr(gpu_budget, "BUDGET", 1.0)
    gpu_budget.guard("small job", 60, "A10G")  # about $0.03: fits in $1
    with pytest.raises(SystemExit):
        gpu_budget.guard("huge job", 100_000, "A100-40GB")  # about $60: refused before anything launches


def test_the_ledger_counts_toward_the_remaining_budget(tmp_path, monkeypatch):
    monkeypatch.setattr(gpu_budget, "LEDGER", tmp_path / "spend.json")
    monkeypatch.setattr(gpu_budget, "BUDGET", 1.0)
    gpu_budget.ledger({"est_cost_usd": 0.99})
    assert gpu_budget.spent() == pytest.approx(0.99)
    with pytest.raises(SystemExit):
        gpu_budget.guard("job after spending", 600, "A10G")  # about $0.2 against the $0.01 left
