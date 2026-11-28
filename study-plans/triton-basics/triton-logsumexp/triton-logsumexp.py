import torch
import triton
import triton.language as tl


@triton.jit
def logsumexp_kernel(x_ptr, out_ptr, x_row_stride, n_cols, BLOCK_SIZE: tl.constexpr):
    # Write code here
    pid = tl.program_id(0)
    x_row_start = x_ptr + pid * x_row_stride
    offset = tl.arange(0, BLOCK_SIZE)
    mask = offset < n_cols

    row = tl.load(x_row_start + offset, mask=mask, other=-float("inf"))

    m = tl.max(row, axis=0)
    new_row = tl.exp(row - m)
    sum = tl.sum(new_row)
    output = m + tl.log(sum)
 
    tl.store(out_ptr + pid, output)

def solve(x: torch.Tensor, out: torch.Tensor) -> None:
    """Launch logsumexp_kernel with one program per row."""
    M, N = x.shape
    BLOCK_SIZE = triton.next_power_of_2(N)
    grid = (M,)
    logsumexp_kernel[grid](
        x, out, x.stride(0), N, BLOCK_SIZE=BLOCK_SIZE,
    )