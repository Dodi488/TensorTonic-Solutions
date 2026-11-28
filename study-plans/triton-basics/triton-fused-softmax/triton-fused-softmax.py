import torch
import triton
import triton.language as tl


@triton.jit
def softmax_kernel(x_ptr, out_ptr, x_row_stride, out_row_stride, n_cols, BLOCK_SIZE: tl.constexpr):
    # Write code here
    idx = tl.program_id(0)
    x_row_start = x_ptr + idx * x_row_stride
    offset = tl.arange(0, BLOCK_SIZE)
    mask = offset < n_cols

    row = tl.load(x_row_start + offset, mask=mask, other=-float("inf"))

    new_row = row - tl.max(row, axis=0)
    denominator = tl.exp(new_row)
    numerator = tl.sum(denominator, axis=0)
    output = denominator / numerator

    out_row_start = out_ptr + idx * out_row_stride
    tl.store(out_row_start + offset, output, mask=mask)

def solve(x: torch.Tensor, out: torch.Tensor) -> None:
    """Launch softmax_kernel with one program per row."""
    M, N = x.shape
    BLOCK_SIZE = triton.next_power_of_2(N)
    grid = (M,)
    softmax_kernel[grid](
        x, out, x.stride(0), out.stride(0), N, BLOCK_SIZE=BLOCK_SIZE,
    )