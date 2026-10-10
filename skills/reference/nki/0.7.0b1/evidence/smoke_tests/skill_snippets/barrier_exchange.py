# example: barrier_exchange
import nki
import nki.language as nl
import nki.isa as nisa


@nki.jit
def barrier_exchange(dummy):
    # each core writes its own row of a shared_hbm scratch, waits, then reads both rows
    out = nl.ndarray((2, 2, 64), dtype=nl.float32, buffer=nl.shared_hbm)
    scratch = nl.ndarray((2, 64), dtype=nl.float32, buffer=nl.shared_hbm)   # internal scratch, not returned
    pid = nl.program_id(0)
    mine = nl.ndarray((1, 64), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=mine, value=float(pid + 1))
    nisa.dma_copy(dst=scratch[pid:pid + 1, :], src=mine)
    nisa.core_barrier(data=scratch, cores=(0, 1))
    both = nl.ndarray((2, 64), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=both, src=scratch)
    nisa.dma_copy(dst=out[pid], src=both)      # out[0] == out[1] == [[1...], [2...]]
    return out
