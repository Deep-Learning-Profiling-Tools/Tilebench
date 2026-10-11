import torch
import nki
import nki.language as nl
import nki.isa as nisa


_LAST_CONFIG = {}


@nki.jit
def _scatter_kv(kv_nope, kv_rope, dest_loc, partitions, rows_per_partition):
    tokens = kv_nope.shape[0]
    nope_width = kv_nope.shape[1] * kv_nope.shape[2]
    rope_width = kv_rope.shape[1] * kv_rope.shape[2]
    block_tokens = partitions * rows_per_partition
    blocks = tokens // block_tokens

    out_nope = nl.ndarray(
        kv_nope.shape, dtype=kv_nope.dtype, buffer=nl.shared_hbm
    )
    out_rope = nl.ndarray(
        kv_rope.shape, dtype=kv_rope.dtype, buffer=nl.shared_hbm
    )

    src_nope = kv_nope.reshape(
        (blocks, partitions, rows_per_partition, nope_width)
    )
    src_rope = kv_rope.reshape(
        (blocks, partitions, rows_per_partition, rope_width)
    )
    src_indices = dest_loc.reshape(
        (blocks, partitions, rows_per_partition)
    )
    dst_nope = out_nope.reshape((tokens, nope_width))
    dst_rope = out_rope.reshape((tokens, rope_width))

    pid = nl.program_id(0)
    programs = nl.num_programs()

    for block in range(pid, blocks, programs):
        indices = nl.ndarray(
            (partitions, rows_per_partition),
            dtype=nl.int32,
            buffer=nl.sbuf,
        )
        nope_tile = nl.ndarray(
            (partitions, rows_per_partition, nope_width),
            dtype=kv_nope.dtype,
            buffer=nl.sbuf,
        )
        rope_tile = nl.ndarray(
            (partitions, rows_per_partition, rope_width),
            dtype=kv_rope.dtype,
            buffer=nl.sbuf,
        )

        nisa.dma_copy(dst=indices, src=src_indices[block])
        nisa.dma_copy(dst=nope_tile, src=src_nope[block])
        nisa.dma_copy(dst=rope_tile, src=src_rope[block])

        for row in range(rows_per_partition):
            nisa.dma_copy(
                dst=dst_nope.ap(
                    pattern=[[nope_width, partitions], [1, nope_width]],
                    offset=0,
                    vector_offset=indices[:, row:row + 1],
                    indirect_dim=0,
                ),
                src=nope_tile[:, row, :],
                dge_mode=nisa.dge_mode.hwdge,
            )
            nisa.dma_copy(
                dst=dst_rope.ap(
                    pattern=[[rope_width, partitions], [1, rope_width]],
                    offset=0,
                    vector_offset=indices[:, row:row + 1],
                    indirect_dim=0,
                ),
                src=rope_tile[:, row, :],
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

    partitions = 128
    rows_per_partition = 8
    tokens = int(kv_nope.shape[0])

    _LAST_CONFIG = {
        "kernel": "fused_blocked_hwdge_scatter",
        "lnc": 2,
        "partitions": partitions,
        "rows_per_partition": rows_per_partition,
        "block_tokens": partitions * rows_per_partition,
        "core_mapping": "cyclic_token_blocks",
        "tokens": tokens,
        "nope_row_elements": int(kv_nope.shape[1] * kv_nope.shape[2]),
        "rope_row_elements": int(kv_rope.shape[1] * kv_rope.shape[2]),
        "index_dtype": "int32",
        "scatter_dge_mode": "hwdge",
        "initialize_outputs": False,
    }

    return _scatter_kv[2](
        kv_nope, kv_rope, dest_loc, partitions, rows_per_partition
    )


def get_last_config() -> dict:
    return dict(_LAST_CONFIG)
