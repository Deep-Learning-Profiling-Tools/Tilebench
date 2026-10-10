"""(b) LNC=2 launch semantics on this host (NEURON_LOGICAL_NC_CONFIG=2).

probe kernel: each program writes one row of a shared_hbm output with
   value = 100 * program_id + 10 * num_programs + program_ndim
   (SBUF tile allocated identically on both cores, filled with a per-core value:
   if SBUF were shared between the two physical cores the rows would mix).
split kernel: each program handles half of the rows of a [256, 512] add.
"""

import _common as C
from _common import nki, nl, nisa, torch


@nki.jit
def probe(dummy):
    out = nl.ndarray((2, 8), dtype=nl.float32, buffer=nl.shared_hbm)
    pid = nl.program_id(0)
    n = nl.num_programs()
    nd = nl.program_ndim()
    t = nl.ndarray((1, 8), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=t, value=100 * pid + 10 * n + nd)
    nisa.dma_copy(dst=out[pid:pid + 1, :], src=t)
    return out


@nki.jit
def split_add(a, b):
    out = nl.ndarray(a.shape, dtype=a.dtype, buffer=nl.shared_hbm)
    P = 128
    n_prog = nl.num_programs()          # 2 under LNC=2, 1 under kernel[1]
    pid = nl.program_id(0)              # 0 or 1
    rows_per_prog = a.shape[0] // n_prog
    for i in range(rows_per_prog // P):
        r0 = pid * rows_per_prog + i * P
        a_t = nl.ndarray((P, a.shape[1]), dtype=a.dtype, buffer=nl.sbuf)
        b_t = nl.ndarray((P, a.shape[1]), dtype=a.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=a_t, src=a[r0:r0 + P, :])
        nisa.dma_copy(dst=b_t, src=b[r0:r0 + P, :])
        nisa.tensor_tensor(dst=a_t, data1=a_t, data2=b_t, op=nl.add)   # in place is allowed
        nisa.dma_copy(dst=out[r0:r0 + P, :], src=a_t)
    return out


@nki.jit
def pid_python_type(dummy):
    # Is program_id a trace-time Python int? (usable in Python `if`, slicing, range)
    out = nl.ndarray((1, 4), dtype=nl.float32, buffer=nl.shared_hbm)
    t = nl.ndarray((1, 4), dtype=nl.float32, buffer=nl.sbuf)
    if nl.program_id(0) == 0:
        nisa.memset(dst=t, value=7.0)
        nisa.dma_copy(dst=out, src=t)
    return out


dummy = C.to_dev(torch.zeros(1, 8, dtype=torch.float32))


def run_probe(launch):
    out = launch(dummy)
    C.sync()
    v = out.cpu()
    return v[:, 0].tolist()


def probe_bare():
    rows = run_probe(probe)
    return {"ok": rows == [21.0, 121.0], "row_values": rows,
            "decoded": "row r = 100*program_id + 10*num_programs + program_ndim"}


def probe_lnc2():
    rows = run_probe(probe[2])
    return {"ok": rows == [21.0, 121.0], "row_values": rows}


def probe_lnc1():
    rows = run_probe(probe[1])
    # LNC=1: program_id()==0, num_programs()==1, program_ndim()==0 -> row 0 == 10.
    # Row 1 is never written (uninitialised shared_hbm output; may show stale data).
    return {"ok": rows[0] == 10.0, "row_values": rows,
            "row1_note": "row 1 not written under LNC=1 (uninitialised memory, value meaningless)"}


torch.manual_seed(1)
a_cpu = torch.randn(256, 512, dtype=torch.float16)
b_cpu = torch.randn(256, 512, dtype=torch.float16)
ref = a_cpu.float() + b_cpu.float()
a, b = C.to_dev(a_cpu), C.to_dev(b_cpu)


def split_bare():
    out = split_add(a, b)
    C.sync()
    ok, err = C.allclose(out, ref, 1e-3, 1e-3)
    return {"ok": ok, "max_abs_err": err}


def split_lnc1():
    out = split_add[1](a, b)
    C.sync()
    ok, err = C.allclose(out, ref, 1e-3, 1e-3)
    return {"ok": ok, "max_abs_err": err}


def pid_if():
    out = pid_python_type(dummy)
    C.sync()
    return {"ok": float(out.cpu()[0, 0]) == 7.0, "value": float(out.cpu()[0, 0])}


C.run_case("probe_bare_call", "nl.program_id / nl.num_programs / nl.program_ndim",
           "bare kernel(...) under NEURON_LOGICAL_NC_CONFIG=2: both programs run, num_programs()==2, program_ndim()==1; SBUF private per core", probe_bare)
C.run_case("probe_lnc2_bracket", "kernel[2](...)", "same values as bare call", probe_lnc2)
C.run_case("probe_lnc1_bracket", "kernel[1](...)", "num_programs()==1, program_id()==0, program_ndim()==0; only one program runs; no runtime error although NEURON_LOGICAL_NC_CONFIG=2", probe_lnc1)
C.run_case("split_rows_bare", "program_id row split", "each core handles half the rows (bare call = LNC=2); correctness", split_bare)
C.run_case("split_rows_lnc1", "program_id row split under kernel[1]", "same kernel with one program covers all rows; correctness", split_lnc1)
C.run_case("program_id_in_python_if", "nl.program_id in Python if", "program_id is a trace-time value usable in a Python if/slice", pid_if)
C.emit()
