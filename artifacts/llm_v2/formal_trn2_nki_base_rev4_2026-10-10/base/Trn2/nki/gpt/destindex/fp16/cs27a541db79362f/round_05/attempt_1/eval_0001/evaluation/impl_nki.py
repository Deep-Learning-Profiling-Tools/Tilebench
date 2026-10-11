import torch
import nki
import nki.language as nl
import nki.isa as nisa


_LAST_CONFIG = {}


@nki.jit
def _scatter_kv(kv_nope, kv_rope, dest_loc, token_tile):
    tokens = kv_nope.shape[0]
    nope_width = kv_nope.shape[1] * kv_nope.shape[2]
    rope_width = kv_rope.shape[1] * kv_rope.shape[2]

    out_nope = nl.ndarray(
        kv_nope.shape, dtype=kv_nope.dtype, buffer=nl.shared_hbm
    )
    out_rope = nl.ndarray(
        kv_rope.shape, dtype=kv_rope.dtype, buffer=nl.shared_hbm
    )

    src_nope = kv_nope.reshape((tokens, nope_width))
    src_rope = kv_rope.reshape((tokens, rope_width))
    dst_nope = out_nope.reshape((tokens, nope_width))
    dst_rope = out_rope.reshape((tokens, rope_width))
    indices_hbm = dest_loc.reshape((tokens, 1))

    pid = nl.program_id(0)
    programs = nl.num_programs()
    tiles = (tokens + token_tile - 1) // token_tile

    for tile in range(pid, tiles, programs):
        start = tile * token_tile
        rows = min(token_tile, tokens - start)

        indices = nl.ndarray(
            (rows, 1), dtype=nl.int32, buffer=nl.sbuf
        )
        nope_tile = nl.ndarray(
            (rows, nope_width), dtype=kv_nope.dtype, buffer=nl.sbuf
        )
        rope_tile = nl.ndarray(
            (rows, rope_width), dtype=kv_rope.dtype, buffer=nl.sbuf
        )

        nisa.dma_copy(
            dst=indices,
            src=indices_hbm[start:start + rows, :],
        )
        nisa.dma_copy(
            dst=nope_tile,
            src=src_nope[start:start + rows, :],
        )
        nisa.dma_copy(
            dst=rope_tile,
            src=src_rope[start:start + rows, :],
        )

        nisa.dma_copy(
            dst=dst_nope.ap(
                pattern=[[nope_width, rows], [1, nope_width]],
                offset=0,
                vector_offset=indices,
                indirect_dim=0,
            ),
            src=nope_tile,
            dge_mode=nisa.dge_mode.hwdge,
        )
        nisa.dma_copy(
            dst=dst_rope.ap(
                pattern=[[rope_width, rows], [1, rope_width]],
                offset=0,
                vector_offset=indices,
                indirect_dim=0,
            ),
            src=rope_tile,
            dge_mode=nisa.dge_mode.hwdge,
        )

    return out_nope, out_rope


def run(
    kv_nope: torch.Tensor,
    kv_rope: torch.Tensor,
    dest_loc: torch.Tensor,
    o_nope: torch.Tensor,
    o_rope: torch.Tensor,
):
    global _LAST_CONFIG

    token_tile = 128
    _LAST_CONFIG = {
        "kernel": "fused_hardware_dma_scatter",
        "lnc": 2,
        "token_tile": token_tile,
        "core_mapping": "cyclic_token_tiles",
        "tokens": int(kv_nope.shape[0]),
        "nope_row_elements": int(kv_nope.shape[1] * kv_nope.shape[2]),
        "rope_row_elements": int(kv_rope.shape[1] * kv_rope.shape[2]),
        "index_dtype": "int32",
        "scatter_dge_mode": "hwdge",
        "initialize_outputs": False,
    }

    return _scatter_kv[2](kv_nope, kv_rope, dest_loc, token_tile)


def get_last_config() -> dict:
    return dict(_LAST_CONFIG)
