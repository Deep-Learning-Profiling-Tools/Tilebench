# 2D Convolution (Forward)

Implement a program that performs a 2D convolution between a batched 4D input tensor and a 4D weight tensor, with optional padding, stride, and grouping. The semantics match PyTorch's `torch.nn.functional.conv2d` with `bias=None`.

The input consists of:

- `input`: A 4D array of shape $(batch, in\_channels, H, W)$.
- `weight`: A 4D array of shape $(out\_channels, in\_channels / groups, kernel\_size, kernel\_size)$.
- `stride`: An int (default 1) — convolution stride in both spatial dimensions.
- `padding`: An int (default 1) — zero-padding added to each side.
- `groups`: An int (default 1) — splits input/output channels into independent groups.

The output should be written to the `output` array of shape $(batch, out\_channels, H_{out}, W_{out})$ where

$$
H_{out} = \lfloor (H + 2 \cdot padding - kernel\_size) / stride \rfloor + 1, \quad W_{out} = \lfloor (W + 2 \cdot padding - kernel\_size) / stride \rfloor + 1.
$$

The operation is defined mathematically (for one batch element $b$, one output channel $oc$, and one output position $(oh, ow)$) as:

$$
output[b, oc, oh, ow] = \sum_{ic=0}^{in\_channels/groups - 1} \sum_{kh=0}^{kernel\_size-1} \sum_{kw=0}^{kernel\_size-1} input[b, g \cdot \tfrac{in\_channels}{groups} + ic, ih, iw] \cdot weight[oc, ic, kh, kw]
$$

where $g = oc \,/\, (out\_channels / groups)$, $ih = oh \cdot stride + kh - padding$, $iw = ow \cdot stride + kw - padding$. Out-of-bounds $(ih, iw)$ contribute $0$ (zero padding).

## Implementation Requirements

- Use only native features (external libraries are not permitted)
- The `solve` function signature must remain unchanged
- The final result must be stored in the array `output`
- Bias is **not** added (the framework's `bias=None`)
- Groups dimension must be supported (typically 1)
- Tensor cores (e.g. via implicit GEMM) are recommended for performance

## Example 1

Simple case: batch=1, in_channels=1, out_channels=1, H=W=3, kernel=3, stride=1, padding=1, groups=1, all-ones weight, identity input:

```text
input  = [[[[1, 0, 0], [0, 1, 0], [0, 0, 1]]]]
weight = [[[[1, 1, 1], [1, 1, 1], [1, 1, 1]]]]
With padding=1 the convolution sums each 3x3 neighbourhood of input.
Output ≈ [[[[2, 2, 1], [2, 3, 2], [1, 2, 2]]]]
```

## Constraints

- $batch \geq 1$, $in\_channels \geq 1$, $out\_channels \geq 1$, $H \geq 1$, $W \geq 1$
- $kernel\_size \geq 1$
- $in\_channels$ is divisible by $groups$, $out\_channels$ is divisible by $groups$
- $stride \geq 1$, $padding \geq 0$
