"""Fresh-input and anti-cache verification helpers (device-agnostic torch).

Oracle handling: the reference output is frozen (cloned recursively) BEFORE
the inputs are restored, so a reference that updates an input in place and
returns that input cannot be overwritten by the restore; the candidate's
output is frozen the same way before comparison. Mutation is judged per
input index against the contract's declared mutable set: an undeclared
mutation fails the check, a declared one is restored before every call.
Output aliasing with any input is detected and reported (the contracts
require fresh outputs unless they say otherwise)."""
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


def freeze(out):
    """Recursive clone of every tensor in an output structure."""
    if isinstance(out, torch.Tensor):
        return out.detach().clone()
    if isinstance(out, (tuple, list)):
        return type(out)(freeze(o) for o in out)
    if isinstance(out, dict):
        return {k: freeze(v) for k, v in out.items()}
    return out


def _storage_ptrs(out) -> set[int]:
    if isinstance(out, torch.Tensor):
        try:
            return {out.untyped_storage().data_ptr()}
        except Exception:  # noqa: BLE001
            return {out.data_ptr()}
    if isinstance(out, (tuple, list)):
        s: set[int] = set()
        for o in out:
            s |= _storage_ptrs(o)
        return s
    return set()


def output_aliases(out, inputs: tuple) -> list[int]:
    """Indices of inputs whose storage is shared by an output tensor."""
    ptrs = _storage_ptrs(out)
    hits = []
    for i, x in enumerate(inputs):
        if isinstance(x, torch.Tensor) and _storage_ptrs(x) & ptrs:
            hits.append(i)
    return hits


@dataclass
class CheckOutcome:
    name: str
    ok: bool
    message: str = ""
    inputs_mutated: list[int] = field(default_factory=list)
    output_aliases_inputs: list[int] = field(default_factory=list)

    def to_dict(self) -> dict:
        return dict(self.__dict__)


def run_numerical_checks(run_candidate, run_reference, make_inputs, *, atol: float, rtol: float,
                         mutable_indices: set[int] | frozenset[int] = frozenset(), sync,
                         aliasing_allowed: bool = False) -> list[CheckOutcome]:
    """Three checks, each a fresh evaluation:
    1. fresh_storage: brand-new input tensors (new values, new addresses)
    2. same_address_new_values: refill the first check's tensors in place
    3. repeat_same_inputs: run twice on identical inputs; the second output must
       still match (detects state carried between calls / input corruption)
    Inputs listed in `mutable_indices` may be modified by the candidate and are
    restored before every call; any other mutation fails the check."""
    out: list[CheckOutcome] = []
    mutable = set(mutable_indices)

    def once(name: str, inputs: tuple) -> tuple:
        snap = snapshot(inputs)
        ref = freeze(run_reference(*inputs))
        sync()
        restore(inputs, snap)
        cand_raw = run_candidate(*inputs)
        sync()
        aliases = output_aliases(cand_raw, inputs)
        cand = freeze(cand_raw)
        mutated = inputs_changed(inputs, snap)
        ok, msg = verify(cand, ref, atol=atol, rtol=rtol)
        undeclared = [i for i in mutated if i not in mutable]
        if undeclared:
            ok, msg = False, f"candidate mutated input(s) {undeclared} which the contract declares immutable; {msg}"
        if aliases and not aliasing_allowed:
            ok, msg = False, f"candidate output shares storage with input(s) {aliases}; the contract requires a fresh output; {msg}"
        restore(inputs, snap)
        out.append(CheckOutcome(name=name, ok=ok, message=msg, inputs_mutated=mutated, output_aliases_inputs=aliases))
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
