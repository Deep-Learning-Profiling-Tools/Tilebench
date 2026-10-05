"""Fresh-input and anti-cache verification helpers (device-agnostic torch)."""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

from tilebench.core.verifier import verify


def snapshot(inputs: tuple) -> list:
    return [x.clone() if isinstance(x, torch.Tensor) else x for x in inputs]


def restore(inputs: tuple, snap: list) -> None:
    for x, s in zip(inputs, snap):
        if isinstance(x, torch.Tensor):
            x.copy_(s)


def inputs_changed(inputs: tuple, snap: list) -> list[int]:
    changed = []
    for i, (x, s) in enumerate(zip(inputs, snap)):
        if isinstance(x, torch.Tensor) and not torch.equal(x, s):
            changed.append(i)
    return changed


def refill_in_place(inputs: tuple, fresh: tuple) -> None:
    """Write fresh values into the SAME storage (same data_ptr): a candidate
    that keys a cache on tensor identity/address must still recompute."""
    for x, y in zip(inputs, fresh):
        if isinstance(x, torch.Tensor):
            x.copy_(y)


@dataclass
class CheckOutcome:
    name: str
    ok: bool
    message: str = ""
    inputs_mutated: list[int] = field(default_factory=list)

    def to_dict(self) -> dict:
        return self.__dict__


def run_numerical_checks(run_candidate, run_reference, make_inputs, *, atol: float, rtol: float,
                         restore_required: bool, sync) -> list[CheckOutcome]:
    """Three checks, each a fresh evaluation:
    1. fresh_storage: brand-new input tensors (new values, new addresses)
    2. same_address_new_values: refill the first check's tensors in place
    3. repeat_same_inputs: run twice on identical inputs; the second output must
       still match (detects state carried between calls / input corruption)
    Mutated inputs are detected via snapshots; when the contract declares the
    operator mutates an input, the snapshot is restored before every call."""
    out: list[CheckOutcome] = []

    def once(name: str, inputs: tuple) -> tuple:
        snap = snapshot(inputs)
        ref = run_reference(*inputs)
        sync()
        restore(inputs, snap)
        cand = run_candidate(*inputs)
        sync()
        mutated = inputs_changed(inputs, snap)
        ok, msg = verify(cand, ref, atol=atol, rtol=rtol)
        if mutated and not restore_required:
            ok, msg = False, f"candidate mutated input(s) {mutated} which the contract declares immutable; {msg}"
        restore(inputs, snap)
        out.append(CheckOutcome(name=name, ok=ok, message=msg, inputs_mutated=mutated))
        return inputs

    a = once("fresh_storage", make_inputs())
    if not out[-1].ok:
        return out
    refill_in_place(a, make_inputs())
    once("same_address_new_values", a)
    if not out[-1].ok:
        return out
    once("repeat_same_inputs", a)
    return out
