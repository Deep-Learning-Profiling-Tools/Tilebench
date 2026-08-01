import torch

try:
    import neuronxcc.nki as nki
    import neuronxcc.nki.language as nl
    import neuronxcc.nki.isa as nisa
    PMAX = nl.tile_size.pmax
except ImportError:
    nki = None

if nki is not None:
    @nki.jit
    def cross_entropy_kernel(logits, targets):
        B, C = logits.shape
        NEG_INF = -3.0e38

        num_blocks = (B + (PMAX - 1)) // PMAX
        hbm_result = nl.ndarray((B, 1), dtype=logits.dtype, buffer=nl.hbm)

        for i in range(num_blocks):
            offset = i * PMAX
            partition_index = nl.arange(PMAX)[:, None]
            row_mask = partition_index < (B - offset)

            class_index = nl.arange(C)[None, :]
            zero_index = nl.arange(1)[None, :]

            logits_tile = nl.load(
                logits[offset + partition_index, class_index],
                mask=row_mask, dtype=nl.float32,
            )
            targets_tile = nl.load(
                targets[offset + partition_index, zero_index],
                mask=row_mask,
            )

            grid = nl.mgrid[0:PMAX, 0:C]
            class_iota = nisa.iota(expr=grid.x, dtype=nl.int32)

            neg_fill = nl.full(logits_tile.shape, NEG_INF, dtype=nl.float32, buffer=nl.sbuf)
            logits_for_max = nl.where(row_mask, logits_tile, neg_fill)
            row_max = nl.max(logits_for_max, axis=1, keepdims=True)

            shifted = nl.subtract(logits_tile, row_max)
            exp_tile = nl.exp(shifted)
            zero_fill = nl.zeros(exp_tile.shape, dtype=nl.float32, buffer=nl.sbuf)
            exp_safe = nl.where(row_mask, exp_tile, zero_fill)
            row_sum = nl.sum(exp_safe, axis=1, keepdims=True)
            log_sum = nl.log(row_sum)

            target_mask = class_iota == targets_tile
            zero_fill2 = nl.zeros(shifted.shape, dtype=nl.float32, buffer=nl.sbuf)
            target_shifted_full = nl.where(target_mask, shifted, zero_fill2)
            target_shifted = nl.sum(target_shifted_full, axis=1, keepdims=True)

            loss = nl.subtract(log_sum, target_shifted)
            loss_out = nl.add(loss, 0.0, dtype=logits.dtype)

            nl.store(hbm_result[offset + partition_index, zero_index],
                     value=loss_out, mask=row_mask)

        return hbm_result


def run(logits: torch.Tensor, targets: torch.Tensor, block_size: int = 1024,
        autotune: bool = False, **kwargs) -> torch.Tensor:
    targets_2d = targets.reshape(-1, 1).to(torch.int32)
    result = cross_entropy_kernel(logits, targets_2d)
    return result.reshape(-1)


def get_last_config() -> dict | None:
    return None
