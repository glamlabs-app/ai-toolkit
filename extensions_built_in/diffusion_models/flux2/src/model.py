import threading

import torch
from einops import rearrange
from torch import Tensor, nn
import torch.utils.checkpoint as ckpt
from torch.utils.checkpoint import CheckpointPolicy, create_selective_checkpoint_contexts
from toolkit.offloaded_checkpoint import offloaded_checkpoint
import math
import os
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Selective Activation Checkpointing (SAC) policy
# ---------------------------------------------------------------------------
# Save outputs of expensive ops (matmuls, attention) so they aren't recomputed
# during backward.  Recompute cheap ops (norms, activations, element-wise).
_SAC_SAVE_OPS = frozenset({
    torch.ops.aten.mm.default,
    torch.ops.aten.addmm.default,
    torch.ops.aten.bmm.default,
    torch.ops.aten._scaled_mm.default,
    torch.ops.aten._scaled_dot_product_flash_attention.default,
    torch.ops.aten._scaled_dot_product_efficient_attention.default,
    torch.ops.aten._scaled_dot_product_cudnn_attention.default,
})


def _sac_policy_fn(ctx, op, *args, **kwargs):
    if op in _SAC_SAVE_OPS:
        return CheckpointPolicy.MUST_SAVE
    return CheckpointPolicy.PREFER_RECOMPUTE


# ---------------------------------------------------------------------------
# Recompute cache for non-aten ops (fouroversix fp4_matmul, etc.)
# ---------------------------------------------------------------------------
# Custom C++ extension ops are invisible to the SAC policy, so their outputs
# would be recomputed during checkpoint replay.  This FIFO cache stores their
# outputs during the initial forward and returns them during replay, skipping
# the expensive kernel re-execution.
#
# Safe because _FourOverSixLinearFn.backward only needs grad_output + frozen
# weights — it never uses the forward input or output value.
class _NonAtenRecomputeCache:
    _local = threading.local()

    def __init__(self):
        self.outputs: list[Tensor] = []
        self.idx: int = 0
        self.mode: str | None = None  # 'save' or 'recompute'

    @classmethod
    def active(cls) -> "_NonAtenRecomputeCache | None":
        return getattr(cls._local, "inst", None)

    @classmethod
    def _install(cls, inst: "_NonAtenRecomputeCache | None"):
        cls._local.inst = inst

    def save_output(self, out: Tensor):
        if self.mode == "save":
            self.outputs.append(out.detach())

    def get_cached(self) -> Tensor | None:
        if self.mode == "recompute" and self.idx < len(self.outputs):
            out = self.outputs[self.idx]
            self.idx += 1
            return out
        return None


def _sac_context_fn():
    save_ctx, recompute_ctx = create_selective_checkpoint_contexts(_sac_policy_fn)
    cache = _NonAtenRecomputeCache()

    class _SaveContext:
        def __enter__(self_):
            cache.mode = "save"
            cache.outputs.clear()
            cache.idx = 0
            _NonAtenRecomputeCache._install(cache)
            return save_ctx.__enter__()

        def __exit__(self_, *args):
            cache.mode = None
            _NonAtenRecomputeCache._install(None)
            return save_ctx.__exit__(*args)

    class _RecomputeContext:
        def __enter__(self_):
            cache.mode = "recompute"
            cache.idx = 0
            _NonAtenRecomputeCache._install(cache)
            return recompute_ctx.__enter__()

        def __exit__(self_, *args):
            cache.mode = None
            cache.outputs.clear()
            _NonAtenRecomputeCache._install(None)
            return recompute_ctx.__exit__(*args)

    return _SaveContext(), _RecomputeContext()


@dataclass
class Flux2Params:
    in_channels: int = 128
    context_in_dim: int = 15360
    hidden_size: int = 6144
    num_heads: int = 48
    depth: int = 8
    depth_single_blocks: int = 48
    axes_dim: list[int] = field(default_factory=lambda: [32, 32, 32, 32])
    theta: int = 2000
    mlp_ratio: float = 3.0
    use_guidance_embed: bool = True


@dataclass
class Klein9BParams:
    in_channels: int = 128
    context_in_dim: int = 12288
    hidden_size: int = 4096
    num_heads: int = 32
    depth: int = 8
    depth_single_blocks: int = 24
    axes_dim: list[int] = field(default_factory=lambda: [32, 32, 32, 32])
    theta: int = 2000
    mlp_ratio: float = 3.0
    use_guidance_embed: bool = False


@dataclass
class Klein4BParams:
    in_channels: int = 128
    context_in_dim: int = 7680
    hidden_size: int = 3072
    num_heads: int = 24
    depth: int = 5
    depth_single_blocks: int = 20
    axes_dim: list[int] = field(default_factory=lambda: [32, 32, 32, 32])
    theta: int = 2000
    mlp_ratio: float = 3.0
    use_guidance_embed: bool = False


class FakeConfig:
    # for diffusers compatability
    def __init__(self):
        self.patch_size = 1


def _set_module_by_name(root: nn.Module, module_name: str, new_module: nn.Module):
    if "." in module_name:
        parent_name, leaf_name = module_name.rsplit(".", 1)
        parent = root.get_submodule(parent_name)
    else:
        parent = root
        leaf_name = module_name
    setattr(parent, leaf_name, new_module)


def _run_scaled_mm(
    a_fp8: Tensor,
    b_fp8_t: Tensor,
    scale_a: Tensor,
    scale_b: Tensor,
    out_dtype: torch.dtype,
) -> Tensor:
    if not hasattr(torch, "_scaled_mm"):
        raise RuntimeError(
            "Native FP8 path requested, but torch._scaled_mm is unavailable."
        )
    try:
        out = torch._scaled_mm(
            a_fp8,
            b_fp8_t,
            scale_a=scale_a,
            scale_b=scale_b,
            out_dtype=out_dtype,
        )
    except TypeError:
        out = torch._scaled_mm(a_fp8, b_fp8_t, scale_a, scale_b, out_dtype)
    if isinstance(out, tuple):
        out = out[0]
    return out


_FP8_E5M2_MAX = 57344.0


class _FP8LinearFn(torch.autograd.Function):
    """FP8 scaled matmul with FP8 backward for LoRA training.

    Forward:  FP8 _scaled_mm  (fast tensor-core path)
    Backward: quantize grad_output to FP8 e5m2, compute
              grad_x = grad_output_fp8 @ weight_fp8 via _scaled_mm
              (no grad_weight -- base is frozen)
    """

    @staticmethod
    def forward(
        ctx,
        x_2d: Tensor,
        weight_fp8: Tensor,
        weight_scale: Tensor,
        input_scale: Optional[Tensor],
    ) -> Tensor:
        out_dtype = x_2d.dtype
        w_scale = weight_scale.to(device=x_2d.device, dtype=torch.float32)

        if input_scale is not None:
            s_a = input_scale.to(device=x_2d.device, dtype=torch.float32)
        else:
            max_abs = x_2d.detach().abs().amax()
            s_a = (max_abs / 448.0).clamp(min=1e-8).to(torch.float32)

        x_scaled = (x_2d / s_a).clamp(-448.0, 448.0)
        x_fp8 = x_scaled.to(torch.float8_e4m3fn)
        w_fp8_t = weight_fp8.t()

        out = _run_scaled_mm(x_fp8, w_fp8_t, s_a, w_scale, out_dtype)

        # Build column-major weight for backward _scaled_mm on the fly
        # so we don't permanently double FP8 weight memory.
        # weight_fp8 [N,K] row-major → .t().contiguous().t() → [N,K] column-major
        w_col_major = weight_fp8.t().contiguous().t()
        ctx.save_for_backward(w_col_major, w_scale)
        ctx.out_dtype = out_dtype
        return out

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        w_col_major, w_scale = ctx.saved_tensors

        grad = grad_output.contiguous()
        max_abs = grad.detach().abs().amax()
        grad_scale = (max_abs / _FP8_E5M2_MAX).clamp(min=1e-8).to(torch.float32)
        grad_fp8 = (grad / grad_scale).clamp(-_FP8_E5M2_MAX, _FP8_E5M2_MAX).to(
            torch.float8_e5m2
        )

        grad_x = _run_scaled_mm(
            grad_fp8, w_col_major, grad_scale, w_scale, ctx.out_dtype
        )

        return grad_x, None, None, None


class FP8ScaledLinear(nn.Module):
    """Linear layer backed by FP8 weights + per-layer scales.

    Supports both inference (eager _scaled_mm) and training (custom autograd
    backward that computes grad_x via FP8 _scaled_mm, keeping the full
    forward+backward path in FP8 without dequantizing to BF16).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        weight_fp8: Tensor,
        weight_scale: Tensor,
        input_scale: Optional[Tensor] = None,
        bias: Optional[Tensor] = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer("weight_fp8", weight_fp8, persistent=False)
        self.register_buffer("weight_scale", weight_scale.float(), persistent=False)
        if input_scale is not None:
            self.register_buffer("input_scale", input_scale.float(), persistent=False)
        else:
            self.input_scale = None
        if bias is not None:
            self.register_buffer("bias", bias, persistent=False)
        else:
            self.bias = None
        self._logged_once = False

    def forward(self, x: Tensor) -> Tensor:
        if not x.is_cuda:
            raise RuntimeError(
                "FP8ScaledLinear requires CUDA tensors."
            )

        leading_shape = x.shape[:-1]
        x2d = x.reshape(-1, x.shape[-1]).contiguous()
        out_dtype = x.dtype
        weight_scale = self.weight_scale.to(device=x.device)

        if torch.is_grad_enabled() and x.requires_grad:
            out2d = _FP8LinearFn.apply(
                x2d,
                self.weight_fp8,
                weight_scale,
                self.input_scale,
            )
        else:
            if self.input_scale is not None:
                scale_a = self.input_scale.to(device=x.device)
            else:
                max_abs = x2d.abs().amax()
                scale_a = (max_abs / 448.0).clamp(min=1e-8).to(torch.float32)

            x_fp8 = (x2d / scale_a).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
            w_fp8_t = self.weight_fp8.t()

            if os.getenv("FP8_VERBOSE", "0") == "1" and not self._logged_once:
                print(
                    f"[FP8ScaledLinear] scaled_mm x={tuple(x2d.shape)} w={tuple(self.weight_fp8.shape)} "
                    f"scale_a={float(scale_a)} scale_b={float(weight_scale)}"
                )
                self._logged_once = True

            out2d = _run_scaled_mm(
                x_fp8, w_fp8_t,
                scale_a.to(torch.float32),
                weight_scale.to(torch.float32),
                out_dtype=out_dtype,
            )

        if self.bias is not None:
            out2d = out2d + self.bias.to(device=out2d.device, dtype=out2d.dtype)
        return out2d.view(*leading_shape, self.out_features)


def apply_fp8_checkpoint_to_linears(
    model: nn.Module,
    checkpoint_state_dict: dict[str, Tensor],
) -> int:
    """Replace matching nn.Linear modules with FP8ScaledLinear wrappers."""
    replaced = 0
    named_modules = list(model.named_modules())
    for module_name, module in named_modules:
        if not isinstance(module, nn.Linear):
            continue
        weight_key = f"{module_name}.weight"
        if weight_key not in checkpoint_state_dict:
            continue
        weight_fp8 = checkpoint_state_dict[weight_key]
        if weight_fp8.dtype not in (torch.float8_e4m3fn, torch.float8_e5m2):
            continue

        weight_scale = checkpoint_state_dict.get(f"{module_name}.weight_scale", None)
        if weight_scale is None:
            raise RuntimeError(
                f"Native FP8 checkpoint missing required weight scale for {module_name}."
            )
        input_scale = checkpoint_state_dict.get(f"{module_name}.input_scale", None)
        bias = module.bias.detach().to("cpu") if module.bias is not None else None

        fp8_module = FP8ScaledLinear(
            in_features=module.in_features,
            out_features=module.out_features,
            weight_fp8=weight_fp8.to("cpu"),
            weight_scale=weight_scale.to("cpu"),
            input_scale=input_scale.to("cpu") if input_scale is not None else None,
            bias=bias,
        )
        _set_module_by_name(model, module_name, fp8_module)
        replaced += 1
    return replaced


# ── NVFP4 (Native FP4) support ──────────────────────────────────────────
# Two-level block-scaled 4-bit weights (float4_e2m1fn_x2) from BFL
# FLUX.2 Klein NVFP4 checkpoints.  Forward uses torch._scaled_mm FP4
# tensor-core kernels on Blackwell; backward dequantizes to BF16 for
# grad_x (frozen base, no grad_weight).
# ─────────────────────────────────────────────────────────────────────────

_NVFP4_E2M1_LUT: Optional[Tensor] = None
_NVFP4_QUANT_BOUNDS: Optional[Tensor] = None
_NVFP4_MAX = 6.0

_USE_TRITON_NVFP4: Optional[bool] = None

def _check_triton_nvfp4() -> bool:
    global _USE_TRITON_NVFP4
    if _USE_TRITON_NVFP4 is not None:
        return _USE_TRITON_NVFP4
    if os.getenv("NVFP4_NO_TRITON", "0") == "1":
        _USE_TRITON_NVFP4 = False
        return False
    try:
        import triton  # noqa: F401
        if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 10:
            _USE_TRITON_NVFP4 = True
            return True
    except ImportError:
        pass
    _USE_TRITON_NVFP4 = False
    return False


def _nvfp4_lut(device: torch.device) -> Tensor:
    """16-entry E2M1fn lookup table (lazy, per-device)."""
    global _NVFP4_E2M1_LUT
    if _NVFP4_E2M1_LUT is None or _NVFP4_E2M1_LUT.device != device:
        _NVFP4_E2M1_LUT = torch.tensor(
            # positive                              negative
            [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
             0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
            dtype=torch.bfloat16, device=device,
        )
    return _NVFP4_E2M1_LUT


def _nvfp4_bounds(device: torch.device) -> Tensor:
    """Midpoint boundaries between adjacent E2M1 magnitudes (lazy)."""
    global _NVFP4_QUANT_BOUNDS
    if _NVFP4_QUANT_BOUNDS is None or _NVFP4_QUANT_BOUNDS.device != device:
        _NVFP4_QUANT_BOUNDS = torch.tensor(
            [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0],
            dtype=torch.float32, device=device,
        )
    return _NVFP4_QUANT_BOUNDS


# ── Triton NVFP4 kernels (Blackwell sm_100+) ─────────────────────────────

def _get_triton_nvfp4_kernels():
    """Lazy-import Triton kernels to avoid import cost when not needed."""
    import triton
    import triton.language as tl

    @triton.jit
    def _triton_quantize_nvfp4_kernel(
        x_ptr, packed_ptr, scale_ptr,
        num_blocks,
        TILE: tl.constexpr,
    ):
        """Quantize BF16 activations to NVFP4 (1×16 block scaling).

        Each program handles TILE blocks of 16 values.
        Uses Blackwell cvt.rn.satfinite.e2m1x2.f32 PTX for hardware rounding.
        """
        pid = tl.program_id(0)
        block_ids = pid * TILE + tl.arange(0, TILE)
        mask = block_ids < num_blocks

        base_offsets = block_ids[:, None] * 16 + tl.arange(0, 16)[None, :]
        vals = tl.load(x_ptr + base_offsets, mask=mask[:, None], other=0.0).to(tl.float32)

        amax = tl.max(tl.abs(vals), axis=1)
        amax = tl.maximum(amax, 1e-12)

        scaled = vals * (6.0 / amax)[:, None]

        pairs = scaled.reshape(TILE, 8, 2)
        (lo, hi) = pairs.split()

        packed = tl.inline_asm_elementwise(
            asm="""
            {
            .reg .b8 byte0, byte1, byte2, byte3;
            cvt.rn.satfinite.e2m1x2.f32 byte0, $5, $1;
            cvt.rn.satfinite.e2m1x2.f32 byte1, $6, $2;
            cvt.rn.satfinite.e2m1x2.f32 byte2, $7, $3;
            cvt.rn.satfinite.e2m1x2.f32 byte3, $8, $4;
            mov.b32 $0, {byte0, byte1, byte2, byte3};
            }
            """,
            constraints="=r,r,r,r,r,r,r,r,r",
            args=[lo, hi],
            dtype=tl.uint8,
            is_pure=True,
            pack=4,
        )

        out_offsets = block_ids[:, None] * 8 + tl.arange(0, 8)[None, :]
        tl.store(packed_ptr + out_offsets, packed, mask=mask[:, None])

        scale_vals = (amax / 6.0).to(tl.float8e4nv)
        tl.store(scale_ptr + block_ids, scale_vals, mask=mask)

    @triton.jit
    def _triton_dequant_nvfp4_kernel(
        w_ptr, scale_ptr, scale2_ptr, out_ptr,
        num_rows, in_packed,
        BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr,
    ):
        """Dequantize NVFP4 packed uint8 weights to BF16.

        Each block handles BLOCK_M rows × BLOCK_K packed columns.
        Passes each full byte to cvt.rn.f16x2.e2m1x2 to get both
        lo/hi FP4 values in one instruction.
        """
        pid_m = tl.program_id(0)
        pid_k = tl.program_id(1)

        row_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        col_offs = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)

        row_mask = row_offs < num_rows
        col_mask = col_offs < in_packed
        mask = row_mask[:, None] & col_mask[None, :]

        w_idx = row_offs[:, None] * in_packed + col_offs[None, :]
        w_u32 = tl.load(w_ptr + w_idx, mask=mask, other=0).to(tl.uint32)

        fp16x2 = tl.inline_asm_elementwise(
            asm="cvt.rn.f16x2.e2m1x2 $0, $1;",
            constraints="=r,r",
            args=[w_u32],
            dtype=tl.uint32,
            is_pure=True,
            pack=1,
        )

        lo_bf16 = (fp16x2 & 0xFFFF).to(tl.uint16).to(tl.float16, bitcast=True).to(tl.bfloat16)
        hi_bf16 = ((fp16x2 >> 16) & 0xFFFF).to(tl.uint16).to(tl.float16, bitcast=True).to(tl.bfloat16)

        n_scale_cols = in_packed // 8
        scale_col = col_offs // 8
        scale_idx = row_offs[:, None] * n_scale_cols + scale_col[None, :]
        scale_mask = row_mask[:, None] & (scale_col < n_scale_cols)[None, :]
        scales = tl.load(scale_ptr + scale_idx, mask=scale_mask, other=0).to(tl.bfloat16)

        scale2 = tl.load(scale2_ptr).to(tl.bfloat16)

        lo_val = lo_bf16 * scales * scale2
        hi_val = hi_bf16 * scales * scale2

        K_out = in_packed * 2
        lo_col = col_offs * 2
        hi_col = col_offs * 2 + 1

        lo_idx = row_offs[:, None] * K_out + lo_col[None, :]
        hi_idx = row_offs[:, None] * K_out + hi_col[None, :]

        out_mask = row_mask[:, None] & col_mask[None, :]
        tl.store(out_ptr + lo_idx, lo_val, mask=out_mask)
        tl.store(out_ptr + hi_idx, hi_val, mask=out_mask)

    return _triton_quantize_nvfp4_kernel, _triton_dequant_nvfp4_kernel

_triton_kernels_cache = {}

def _triton_quantize_to_nvfp4(x_2d: Tensor) -> tuple[Tensor, Tensor]:
    """Triton-accelerated NVFP4 quantization using hardware e2m1 conversion."""
    M, K = x_2d.shape
    num_blocks = (M * K) // 16

    packed_flat = torch.empty(num_blocks * 8, dtype=torch.uint8, device=x_2d.device)
    scales = torch.empty(num_blocks, dtype=torch.float8_e4m3fn, device=x_2d.device)

    if 'quant' not in _triton_kernels_cache:
        q_kernel, d_kernel = _get_triton_nvfp4_kernels()
        _triton_kernels_cache['quant'] = q_kernel
        _triton_kernels_cache['dequant'] = d_kernel
    q_kernel = _triton_kernels_cache['quant']

    TILE = 64
    grid = ((num_blocks + TILE - 1) // TILE,)
    q_kernel[grid](x_2d, packed_flat, scales, num_blocks, TILE=TILE)

    x_fp4 = packed_flat.reshape(M, K // 2).view(torch.float4_e2m1fn_x2)
    return x_fp4, scales.contiguous()


def _triton_nvfp4_dequant(
    weight_uint8: Tensor,
    weight_scale: Tensor,
    weight_scale_2: Tensor,
) -> Tensor:
    """Triton-accelerated NVFP4 dequantization using hardware e2m1→fp16 conversion."""
    out_features, in_packed = weight_uint8.shape
    in_features = in_packed * 2
    out = torch.empty(out_features, in_features, dtype=torch.bfloat16, device=weight_uint8.device)

    if 'dequant' not in _triton_kernels_cache:
        q_kernel, d_kernel = _get_triton_nvfp4_kernels()
        _triton_kernels_cache['quant'] = q_kernel
        _triton_kernels_cache['dequant'] = d_kernel
    d_kernel = _triton_kernels_cache['dequant']

    BLOCK_M = 32
    BLOCK_K = min(128, in_packed)
    grid = (
        (out_features + BLOCK_M - 1) // BLOCK_M,
        (in_packed + BLOCK_K - 1) // BLOCK_K,
    )
    d_kernel[grid](
        weight_uint8, weight_scale, weight_scale_2, out,
        out_features, in_packed,
        BLOCK_M=BLOCK_M, BLOCK_K=BLOCK_K,
    )
    return out


# ── Pure-PyTorch fallbacks ────────────────────────────────────────────────

def _nvfp4_dequant(
    weight_uint8: Tensor,
    weight_scale: Tensor,
    weight_scale_2: Tensor,
) -> Tensor:
    """Dequantize NVFP4 packed uint8 weights to BF16.

    Args:
        weight_uint8:  [out_features, in_packed] uint8  (2 fp4 per byte)
        weight_scale:  [out_features, in_packed // 8] float8_e4m3fn
        weight_scale_2: scalar float32  (tensor-level second scale)
    Returns:
        [out_features, in_features] bfloat16
    """
    if _check_triton_nvfp4():
        return _triton_nvfp4_dequant(weight_uint8, weight_scale, weight_scale_2)
    lut = _nvfp4_lut(weight_uint8.device)
    lo = (weight_uint8 & 0x0F).long()
    hi = ((weight_uint8 >> 4) & 0x0F).long()
    w_bf16 = torch.stack([lut[lo], lut[hi]], dim=-1).reshape(
        weight_uint8.shape[0], weight_uint8.shape[1] * 2,
    )
    ws = weight_scale.to(torch.bfloat16).unsqueeze(-1).expand(
        -1, -1, 16,
    ).reshape(weight_uint8.shape[0], weight_uint8.shape[1] * 2)
    return w_bf16 * ws * weight_scale_2.to(torch.bfloat16)


def _quantize_to_nvfp4(x_2d: Tensor) -> tuple[Tensor, Tensor]:
    """Quantize [M, K] BF16 tensor to NVFP4 for ``_scaled_mm``.

    Uses block_size = 16 fp4 values (matching cuBLAS NVFP4 1×16 scaling).

    Returns:
        x_fp4:      [M, K // 2] float4_e2m1fn_x2
        scale_e4m3: 1-D float8_e4m3fn, length M * K // 16
    """
    if _check_triton_nvfp4():
        return _triton_quantize_to_nvfp4(x_2d)
    M, K = x_2d.shape
    x_blocks = x_2d.reshape(-1, 16).float()

    amax = x_blocks.abs().amax(dim=1, keepdim=True).clamp(min=1e-12)
    scale = amax / _NVFP4_MAX
    x_scaled = (x_blocks / scale).clamp(-_NVFP4_MAX, _NVFP4_MAX)

    sign = (x_scaled < 0).to(torch.uint8)
    codes = torch.bucketize(x_scaled.abs(), _nvfp4_bounds(x_2d.device)).to(torch.uint8)
    codes = codes | (sign << 3)

    codes_pairs = codes.reshape(-1, 8, 2)
    packed = (codes_pairs[:, :, 1] << 4) | codes_pairs[:, :, 0]
    x_fp4 = packed.reshape(M, K // 2).view(torch.float4_e2m1fn_x2)

    return x_fp4, scale.squeeze(-1).to(torch.float8_e4m3fn).contiguous()


class _NVFP4LinearFn(torch.autograd.Function):
    """NVFP4 scaled matmul with BF16 backward for LoRA training.

    Forward:  quantize activations to FP4, _scaled_mm  (FP4 tensor cores)
    Backward: dequantize weight to BF16, grad_x = grad @ weight_bf16
              (no grad_weight — base is frozen)
    """

    @staticmethod
    def forward(
        ctx,
        x_2d: Tensor,
        weight_uint8: Tensor,
        weight_scale: Tensor,
        weight_scale_2: Tensor,
    ) -> Tensor:
        out_dtype = x_2d.dtype

        x_fp4, x_scale = _quantize_to_nvfp4(x_2d)
        w_fp4 = weight_uint8.view(torch.float4_e2m1fn_x2)
        w_scale_flat = weight_scale.flatten().contiguous()

        out = torch._scaled_mm(
            x_fp4, w_fp4.t(),
            scale_a=x_scale, scale_b=w_scale_flat,
            out_dtype=out_dtype,
        )
        if isinstance(out, tuple):
            out = out[0]
        out = out * weight_scale_2.to(out_dtype)

        ctx.save_for_backward(weight_uint8, weight_scale, weight_scale_2)
        ctx.out_dtype = out_dtype
        return out

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        weight_uint8, weight_scale, weight_scale_2 = ctx.saved_tensors
        w_bf16 = _nvfp4_dequant(weight_uint8, weight_scale, weight_scale_2)
        grad_x = grad_output.contiguous() @ w_bf16
        return grad_x, None, None, None


class NVFP4ScaledLinear(nn.Module):
    """Linear layer backed by NVFP4 packed weights + two-level block scales.

    Stores weights as uint8 (2 × fp4 per byte) with per-16-element e4m3
    block scales and a per-tensor fp32 second scale.  Forward dispatches to
    FP4 tensor-core ``_scaled_mm`` when CUDA is available, with an automatic
    fallback to dequant-to-BF16 matmul.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        weight_uint8: Tensor,
        weight_scale: Tensor,
        weight_scale_2: Tensor,
        input_scale: Optional[Tensor] = None,
        bias: Optional[Tensor] = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer("weight_uint8", weight_uint8, persistent=False)
        self.register_buffer("weight_scale", weight_scale, persistent=False)
        self.register_buffer("weight_scale_2", weight_scale_2.float(), persistent=False)
        if input_scale is not None:
            self.register_buffer("input_scale", input_scale.float(), persistent=False)
        else:
            self.input_scale = None
        if bias is not None:
            self.register_buffer("bias", bias, persistent=False)
        else:
            self.bias = None
        self._use_fp4_mm = os.getenv("NVFP4_DEQUANT_ONLY", "0") != "1"
        self._logged_once = False

    def forward(self, x: Tensor) -> Tensor:
        leading = x.shape[:-1]
        x2d = x.reshape(-1, x.shape[-1]).contiguous()

        if x.is_cuda and self._use_fp4_mm:
            try:
                out2d = self._forward_fp4(x2d)
            except RuntimeError:
                self._use_fp4_mm = False
                out2d = self._forward_dequant(x2d)
        else:
            out2d = self._forward_dequant(x2d)

        if self.bias is not None:
            out2d = out2d + self.bias.to(device=out2d.device, dtype=out2d.dtype)
        return out2d.view(*leading, self.out_features)

    def _forward_fp4(self, x2d: Tensor) -> Tensor:
        """FP4 tensor-core forward via _scaled_mm."""
        if torch.is_grad_enabled() and x2d.requires_grad:
            return _NVFP4LinearFn.apply(
                x2d, self.weight_uint8, self.weight_scale, self.weight_scale_2,
            )

        x_fp4, x_scale = _quantize_to_nvfp4(x2d)
        w_fp4 = self.weight_uint8.view(torch.float4_e2m1fn_x2)
        w_scale_flat = self.weight_scale.flatten().contiguous()

        if os.getenv("NVFP4_VERBOSE", "0") == "1" and not self._logged_once:
            print(
                f"[NVFP4ScaledLinear] scaled_mm x={tuple(x2d.shape)} "
                f"w_packed={tuple(self.weight_uint8.shape)} "
                f"w_scale_2={float(self.weight_scale_2)}"
            )
            self._logged_once = True

        out = torch._scaled_mm(
            x_fp4, w_fp4.t(),
            scale_a=x_scale, scale_b=w_scale_flat,
            out_dtype=x2d.dtype,
        )
        if isinstance(out, tuple):
            out = out[0]
        return out * self.weight_scale_2.to(x2d.dtype)

    def _forward_dequant(self, x2d: Tensor) -> Tensor:
        """BF16 fallback: dequantize weight on the fly."""
        w_bf16 = _nvfp4_dequant(
            self.weight_uint8, self.weight_scale, self.weight_scale_2,
        )
        return x2d.to(w_bf16.dtype) @ w_bf16.t()


def apply_nvfp4_checkpoint_to_linears(
    model: nn.Module,
    checkpoint_state_dict: dict[str, Tensor],
) -> int:
    """Replace matching nn.Linear modules with NVFP4ScaledLinear wrappers.

    Detects NVFP4 layers by uint8 weight dtype + presence of weight_scale
    and weight_scale_2 keys in the checkpoint.
    """
    replaced = 0
    for module_name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        weight_key = f"{module_name}.weight"
        if weight_key not in checkpoint_state_dict:
            continue
        w = checkpoint_state_dict[weight_key]
        if w.dtype != torch.uint8:
            continue
        scale_key = f"{module_name}.weight_scale"
        scale2_key = f"{module_name}.weight_scale_2"
        if scale_key not in checkpoint_state_dict or scale2_key not in checkpoint_state_dict:
            continue

        ws = checkpoint_state_dict[scale_key]
        ws2 = checkpoint_state_dict[scale2_key]
        inp_s = checkpoint_state_dict.get(f"{module_name}.input_scale", None)
        bias = module.bias.detach().to("cpu") if module.bias is not None else None

        nvfp4_mod = NVFP4ScaledLinear(
            in_features=module.in_features,
            out_features=module.out_features,
            weight_uint8=w.to("cpu"),
            weight_scale=ws.to("cpu"),
            weight_scale_2=ws2.to("cpu"),
            input_scale=inp_s.to("cpu") if inp_s is not None else None,
            bias=bias,
        )
        _set_module_by_name(model, module_name, nvfp4_mod)
        replaced += 1
    return replaced


# ── fouroversix FP4 support ──────────────────────────────────────────────
# Uses the fouroversix library for NVFP4 quantization with 4/6 adaptive
# block scaling.  Provides better quantization quality than standard NVFP4
# round-to-nearest.  Frozen base weights use FP4 tensor-core matmul;
# LoRA adapters stay in full-precision BF16.
# ─────────────────────────────────────────────────────────────────────────


class _FourOverSixLinearFn(torch.autograd.Function):
    """Autograd wrapper: forward and backward both use FP4 tensor-core matmul.

    Forward:  out = fp4_matmul(x, qt_W)      — x @ W^T
    Backward: grad_x = fp4_matmul(g, qt_WT)  — g @ (W^T)^T = g @ W
    No grad_weight — the base model is frozen.
    """

    @staticmethod
    def forward(
        ctx,
        x_2d: Tensor,
        w_values: Tensor, w_scales: Tensor, w_amax: Tensor,
        qt_dtype, qt_original_shape, qt_scale_rule, qt_padded_shape,
        wt_values: Tensor, wt_scales: Tensor, wt_amax: Tensor,
        qt_t_dtype, qt_t_original_shape, qt_t_scale_rule, qt_t_padded_shape,
    ) -> Tensor:
        cache = _NonAtenRecomputeCache.active()
        cached = cache.get_cached() if cache is not None else None
        if cached is not None:
            out = cached
        else:
            from fouroversix import fp4_matmul
            from fouroversix.quantize.quantized_tensor import QuantizedTensor

            qt = QuantizedTensor(
                w_values, w_scales, w_amax,
                qt_dtype, qt_original_shape, qt_scale_rule, qt_padded_shape,
            )
            out = fp4_matmul(x_2d, qt)
            if cache is not None:
                cache.save_output(out)

        ctx.save_for_backward(wt_values, wt_scales, wt_amax)
        ctx.qt_t_dtype = qt_t_dtype
        ctx.qt_t_original_shape = qt_t_original_shape
        ctx.qt_t_scale_rule = qt_t_scale_rule
        ctx.qt_t_padded_shape = qt_t_padded_shape
        return out

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        wt_values, wt_scales, wt_amax = ctx.saved_tensors
        from fouroversix import fp4_matmul
        from fouroversix.quantize.quantized_tensor import QuantizedTensor

        qt_t = QuantizedTensor(
            wt_values, wt_scales, wt_amax,
            ctx.qt_t_dtype, ctx.qt_t_original_shape,
            ctx.qt_t_scale_rule, ctx.qt_t_padded_shape,
        )
        grad_x = fp4_matmul(grad_output.contiguous(), qt_t)
        return (grad_x,) + (None,) * 14


class FourOverSixScaledLinear(nn.Module):
    """Linear layer using fouroversix NVFP4 with 4/6 adaptive block scaling.

    Stores pre-quantized FP4 weights for both W and W^T so that forward
    and backward both run entirely on FP4 tensor cores (no dequantization).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        qt_w,   # QuantizedTensor for W   (forward:  x @ W^T)
        qt_wt,  # QuantizedTensor for W^T (backward: grad @ W)
        bias: Optional[Tensor] = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.register_buffer("_qw_values", qt_w.values, persistent=False)
        self.register_buffer("_qw_scales", qt_w.scale_factors, persistent=False)
        self.register_buffer("_qw_amax", qt_w.amax, persistent=False)
        self._qt_dtype = qt_w.dtype
        self._qt_original_shape = qt_w.original_shape
        self._qt_scale_rule = qt_w.scale_rule
        self._qt_padded_shape = qt_w.padded_shape

        self.register_buffer("_qwt_values", qt_wt.values, persistent=False)
        self.register_buffer("_qwt_scales", qt_wt.scale_factors, persistent=False)
        self.register_buffer("_qwt_amax", qt_wt.amax, persistent=False)
        self._qt_t_dtype = qt_wt.dtype
        self._qt_t_original_shape = qt_wt.original_shape
        self._qt_t_scale_rule = qt_wt.scale_rule
        self._qt_t_padded_shape = qt_wt.padded_shape

        if bias is not None:
            self.register_buffer("bias", bias, persistent=False)
        else:
            self.bias = None

    _PROTECTED_BUFFERS = frozenset({
        "_qw_values", "_qw_scales", "_qw_amax",
        "_qwt_values", "_qwt_scales", "_qwt_amax",
    })

    def _apply(self, fn, recurse=True):
        """Prevent .to(dtype=...) from casting quantized buffers."""
        saved = {k: self._buffers[k] for k in self._PROTECTED_BUFFERS if k in self._buffers}
        result = super()._apply(fn, recurse=recurse)
        for k, v in saved.items():
            self._buffers[k] = v.to(device=self._buffers[k].device)
        return result

    def _make_qt(self):
        from fouroversix.quantize.quantized_tensor import QuantizedTensor
        return QuantizedTensor(
            self._qw_values, self._qw_scales, self._qw_amax,
            self._qt_dtype, self._qt_original_shape,
            self._qt_scale_rule, self._qt_padded_shape,
        )

    def forward(self, x: Tensor) -> Tensor:
        leading = x.shape[:-1]
        x2d = x.reshape(-1, x.shape[-1]).contiguous()

        if torch.is_grad_enabled() and x2d.requires_grad:
            out2d = _FourOverSixLinearFn.apply(
                x2d,
                self._qw_values, self._qw_scales, self._qw_amax,
                self._qt_dtype, self._qt_original_shape,
                self._qt_scale_rule, self._qt_padded_shape,
                self._qwt_values, self._qwt_scales, self._qwt_amax,
                self._qt_t_dtype, self._qt_t_original_shape,
                self._qt_t_scale_rule, self._qt_t_padded_shape,
            )
        else:
            from fouroversix import fp4_matmul
            out2d = fp4_matmul(x2d, self._make_qt())

        if self.bias is not None:
            out2d = out2d + self.bias.to(device=out2d.device, dtype=out2d.dtype)
        return out2d.view(*leading, self.out_features)


def apply_fouroversix_to_linears(
    model: nn.Module,
    skip_patterns: tuple[str, ...] = ("lora",),
) -> int:
    """Replace nn.Linear modules with FourOverSixScaledLinear (NVFP4 4/6).

    Quantizes both W and W^T so forward and backward are both FP4 matmuls
    on Blackwell tensor cores.  Meant to be called **before** LoRA so that
    LoRA adapters sit on top of the quantized base in full-precision BF16.

    Returns the number of replaced modules.
    """
    from fouroversix import quantize_to_fp4

    replaced = 0
    for module_name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if any(p in module_name.lower() for p in skip_patterns):
            continue

        weight = module.weight.data
        if weight.dtype != torch.bfloat16:
            weight = weight.to(torch.bfloat16)
        if not weight.is_cuda:
            weight = weight.cuda()

        qt_w = quantize_to_fp4(weight)
        qt_wt = quantize_to_fp4(weight.t().contiguous())
        bias = module.bias.detach() if module.bias is not None else None

        fos_linear = FourOverSixScaledLinear(
            module.in_features, module.out_features, qt_w, qt_wt, bias,
        )
        _set_module_by_name(model, module_name, fos_linear)
        replaced += 1

    return replaced


class Flux2(nn.Module):
    def __init__(self, params: Flux2Params):
        super().__init__()
        self.config = FakeConfig()

        self.in_channels = params.in_channels
        self.out_channels = params.in_channels
        if params.hidden_size % params.num_heads != 0:
            raise ValueError(
                f"Hidden size {params.hidden_size} must be divisible by num_heads {params.num_heads}"
            )
        pe_dim = params.hidden_size // params.num_heads
        if sum(params.axes_dim) != pe_dim:
            raise ValueError(
                f"Got {params.axes_dim} but expected positional dim {pe_dim}"
            )
        self.hidden_size = params.hidden_size
        self.num_heads = params.num_heads
        self.pe_embedder = EmbedND(
            dim=pe_dim, theta=params.theta, axes_dim=params.axes_dim
        )
        self.img_in = nn.Linear(self.in_channels, self.hidden_size, bias=False)
        self.time_in = MLPEmbedder(
            in_dim=256, hidden_dim=self.hidden_size, disable_bias=True
        )
        self.txt_in = nn.Linear(params.context_in_dim, self.hidden_size, bias=False)

        self.use_guidance_embed = params.use_guidance_embed
        if self.use_guidance_embed:
            self.guidance_in = MLPEmbedder(
                in_dim=256, hidden_dim=self.hidden_size, disable_bias=True
            )

        self.double_blocks = nn.ModuleList(
            [
                DoubleStreamBlock(
                    self.hidden_size,
                    self.num_heads,
                    mlp_ratio=params.mlp_ratio,
                )
                for _ in range(params.depth)
            ]
        )

        self.single_blocks = nn.ModuleList(
            [
                SingleStreamBlock(
                    self.hidden_size,
                    self.num_heads,
                    mlp_ratio=params.mlp_ratio,
                )
                for _ in range(params.depth_single_blocks)
            ]
        )

        self.double_stream_modulation_img = Modulation(
            self.hidden_size,
            double=True,
            disable_bias=True,
        )
        self.double_stream_modulation_txt = Modulation(
            self.hidden_size,
            double=True,
            disable_bias=True,
        )
        self.single_stream_modulation = Modulation(
            self.hidden_size, double=False, disable_bias=True
        )

        self.final_layer = LastLayer(
            self.hidden_size,
            self.out_channels,
        )

        self.gradient_checkpointing = False
        self.selective_checkpointing = False
        self.offload_checkpoint = False
        self.offload_checkpoint_threshold = 0

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def enable_gradient_checkpointing(self):
        self.gradient_checkpointing = True

    def enable_selective_checkpointing(self):
        self.gradient_checkpointing = True
        self.selective_checkpointing = True

    def enable_offload_checkpoint(self, threshold: int = 0):
        """Enable offloaded gradient checkpointing.

        Args:
            threshold: Only offload when img_tokens > threshold.
                0 = always offload. E.g. 2304 for 768px means offload
                only kicks in at resolutions above 768px.
        """
        self.gradient_checkpointing = True
        self.offload_checkpoint = True
        self.offload_checkpoint_threshold = threshold

    def forward(
        self,
        x: Tensor,
        x_ids: Tensor,
        timesteps: Tensor,
        ctx: Tensor,
        ctx_ids: Tensor,
        guidance: Tensor | None,
    ):
        num_txt_tokens = ctx.shape[1]

        timestep_emb = timestep_embedding(timesteps, 256)
        vec = self.time_in(timestep_emb)
        if self.use_guidance_embed:
            guidance_emb = timestep_embedding(guidance, 256)
            vec = vec + self.guidance_in(guidance_emb)

        double_block_mod_img = self.double_stream_modulation_img(vec)
        double_block_mod_txt = self.double_stream_modulation_txt(vec)
        single_block_mod, _ = self.single_stream_modulation(vec)

        img = self.img_in(x)
        txt = self.txt_in(ctx)

        pe_x = self.pe_embedder(x_ids)
        pe_ctx = self.pe_embedder(ctx_ids)

        ckpt_kwargs = dict(use_reentrant=False)
        sac_ctx_fn = _sac_context_fn if self.selective_checkpointing else None
        if sac_ctx_fn is not None:
            ckpt_kwargs["context_fn"] = sac_ctx_fn

        img_tokens = img.shape[1]
        use_offload = (
            self.offload_checkpoint
            and (self.offload_checkpoint_threshold == 0
                 or img_tokens > self.offload_checkpoint_threshold)
        )

        for block in self.double_blocks:
            if torch.is_grad_enabled() and use_offload:
                img, txt = offloaded_checkpoint(
                    block, img, txt, pe_x, pe_ctx,
                    double_block_mod_img, double_block_mod_txt,
                    sac_context_fn=sac_ctx_fn,
                )
            elif torch.is_grad_enabled() and self.gradient_checkpointing:
                img, txt = ckpt.checkpoint(
                    block,
                    img,
                    txt,
                    pe_x,
                    pe_ctx,
                    double_block_mod_img,
                    double_block_mod_txt,
                    **ckpt_kwargs,
                )
            else:
                img, txt = block(
                    img,
                    txt,
                    pe_x,
                    pe_ctx,
                    double_block_mod_img,
                    double_block_mod_txt,
                )

        img = torch.cat((txt, img), dim=1)
        pe = torch.cat((pe_ctx, pe_x), dim=2)

        for i, block in enumerate(self.single_blocks):
            if torch.is_grad_enabled() and use_offload:
                img = offloaded_checkpoint(
                    block, img, pe, single_block_mod,
                    sac_context_fn=sac_ctx_fn,
                )
            elif torch.is_grad_enabled() and self.gradient_checkpointing:
                img = ckpt.checkpoint(
                    block,
                    img,
                    pe,
                    single_block_mod,
                    **ckpt_kwargs,
                )
            else:
                img = block(
                    img,
                    pe,
                    single_block_mod,
                )

        img = img[:, num_txt_tokens:, ...]

        img = self.final_layer(img, vec)
        return img


class SelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=False)

        self.norm = QKNorm(head_dim)
        self.proj = nn.Linear(dim, dim, bias=False)


class SiLUActivation(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_fn = nn.SiLU()

    def forward(self, x: Tensor) -> Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return self.gate_fn(x1) * x2


class Modulation(nn.Module):
    def __init__(self, dim: int, double: bool, disable_bias: bool = False):
        super().__init__()
        self.is_double = double
        self.multiplier = 6 if double else 3
        self.lin = nn.Linear(dim, self.multiplier * dim, bias=not disable_bias)

    def forward(self, vec: torch.Tensor):
        out = self.lin(nn.functional.silu(vec))
        if out.ndim == 2:
            out = out[:, None, :]
        out = out.chunk(self.multiplier, dim=-1)
        return out[:3], out[3:] if self.is_double else None


class LastLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        out_channels: int,
    ):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=False)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=False)
        )

    def forward(self, x: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
        mod = self.adaLN_modulation(vec)
        shift, scale = mod.chunk(2, dim=-1)
        if shift.ndim == 2:
            shift = shift[:, None, :]
            scale = scale[:, None, :]
        x = (1 + scale) * self.norm_final(x) + shift
        x = self.linear(x)
        return x


class SingleStreamBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()

        self.hidden_dim = hidden_size
        self.num_heads = num_heads
        head_dim = hidden_size // num_heads
        self.scale = head_dim**-0.5
        self.mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp_mult_factor = 2

        self.linear1 = nn.Linear(
            hidden_size,
            hidden_size * 3 + self.mlp_hidden_dim * self.mlp_mult_factor,
            bias=False,
        )

        self.linear2 = nn.Linear(
            hidden_size + self.mlp_hidden_dim, hidden_size, bias=False
        )

        self.norm = QKNorm(head_dim)

        self.hidden_size = hidden_size
        self.pre_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        self.mlp_act = SiLUActivation()

    def forward(
        self,
        x: Tensor,
        pe: Tensor,
        mod: tuple[Tensor, Tensor],
    ) -> Tensor:
        mod_shift, mod_scale, mod_gate = mod
        x_mod = (1 + mod_scale) * self.pre_norm(x) + mod_shift

        qkv, mlp = torch.split(
            self.linear1(x_mod),
            [3 * self.hidden_size, self.mlp_hidden_dim * self.mlp_mult_factor],
            dim=-1,
        )

        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        q, k = self.norm(q, k, v)

        attn = attention(q, k, v, pe)

        # compute activation in mlp stream, cat again and run second linear layer
        output = self.linear2(torch.cat((attn, self.mlp_act(mlp)), 2))
        return x + mod_gate * output


class DoubleStreamBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float,
    ):
        super().__init__()
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.num_heads = num_heads
        assert hidden_size % num_heads == 0, (
            f"{hidden_size=} must be divisible by {num_heads=}"
        )

        self.hidden_size = hidden_size
        self.img_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp_mult_factor = 2

        self.img_attn = SelfAttention(
            dim=hidden_size,
            num_heads=num_heads,
        )

        self.img_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim * self.mlp_mult_factor, bias=False),
            SiLUActivation(),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=False),
        )

        self.txt_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_attn = SelfAttention(
            dim=hidden_size,
            num_heads=num_heads,
        )

        self.txt_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_mlp = nn.Sequential(
            nn.Linear(
                hidden_size,
                mlp_hidden_dim * self.mlp_mult_factor,
                bias=False,
            ),
            SiLUActivation(),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=False),
        )

    def forward(
        self,
        img: Tensor,
        txt: Tensor,
        pe: Tensor,
        pe_ctx: Tensor,
        mod_img: tuple[Tensor, Tensor],
        mod_txt: tuple[Tensor, Tensor],
    ) -> tuple[Tensor, Tensor]:
        img_mod1, img_mod2 = mod_img
        txt_mod1, txt_mod2 = mod_txt

        img_mod1_shift, img_mod1_scale, img_mod1_gate = img_mod1
        img_mod2_shift, img_mod2_scale, img_mod2_gate = img_mod2
        txt_mod1_shift, txt_mod1_scale, txt_mod1_gate = txt_mod1
        txt_mod2_shift, txt_mod2_scale, txt_mod2_gate = txt_mod2

        # prepare image for attention
        img_modulated = self.img_norm1(img)
        img_modulated = (1 + img_mod1_scale) * img_modulated + img_mod1_shift

        img_qkv = self.img_attn.qkv(img_modulated)
        img_q, img_k, img_v = rearrange(
            img_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads
        )
        img_q, img_k = self.img_attn.norm(img_q, img_k, img_v)

        # prepare txt for attention
        txt_modulated = self.txt_norm1(txt)
        txt_modulated = (1 + txt_mod1_scale) * txt_modulated + txt_mod1_shift

        txt_qkv = self.txt_attn.qkv(txt_modulated)
        txt_q, txt_k, txt_v = rearrange(
            txt_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads
        )
        txt_q, txt_k = self.txt_attn.norm(txt_q, txt_k, txt_v)

        q = torch.cat((txt_q, img_q), dim=2)
        k = torch.cat((txt_k, img_k), dim=2)
        v = torch.cat((txt_v, img_v), dim=2)

        pe = torch.cat((pe_ctx, pe), dim=2)
        attn = attention(q, k, v, pe)
        txt_attn, img_attn = attn[:, : txt_q.shape[2]], attn[:, txt_q.shape[2] :]

        # calculate the img blocks
        img = img + img_mod1_gate * self.img_attn.proj(img_attn)
        img = img + img_mod2_gate * self.img_mlp(
            (1 + img_mod2_scale) * (self.img_norm2(img)) + img_mod2_shift
        )

        # calculate the txt blocks
        txt = txt + txt_mod1_gate * self.txt_attn.proj(txt_attn)
        txt = txt + txt_mod2_gate * self.txt_mlp(
            (1 + txt_mod2_scale) * (self.txt_norm2(txt)) + txt_mod2_shift
        )
        return img, txt


class MLPEmbedder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, disable_bias: bool = False):
        super().__init__()
        self.in_layer = nn.Linear(in_dim, hidden_dim, bias=not disable_bias)
        self.silu = nn.SiLU()
        self.out_layer = nn.Linear(hidden_dim, hidden_dim, bias=not disable_bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.out_layer(self.silu(self.in_layer(x)))


class EmbedND(nn.Module):
    def __init__(self, dim: int, theta: int, axes_dim: list[int]):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids: Tensor) -> Tensor:
        emb = torch.cat(
            [
                rope(ids[..., i], self.axes_dim[i], self.theta)
                for i in range(len(self.axes_dim))
            ],
            dim=-3,
        )

        return emb.unsqueeze(1)


def timestep_embedding(t: Tensor, dim, max_period=10000, time_factor: float = 1000.0):
    """
    Create sinusoidal timestep embeddings.
    :param t: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an (N, D) Tensor of positional embeddings.
    """
    t = time_factor * t
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, device=t.device, dtype=torch.float32)
        / half
    )

    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    if torch.is_floating_point(t):
        embedding = embedding.to(t)
    return embedding


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor):
        x_dtype = x.dtype
        x = x.float()
        rrms = torch.rsqrt(torch.mean(x**2, dim=-1, keepdim=True) + 1e-6)
        return (x * rrms).to(dtype=x_dtype) * self.scale


class QKNorm(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.query_norm = RMSNorm(dim)
        self.key_norm = RMSNorm(dim)

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        q = self.query_norm(q)
        k = self.key_norm(k)
        return q.to(v), k.to(v)


def attention(q: Tensor, k: Tensor, v: Tensor, pe: Tensor) -> Tensor:
    q, k = apply_rope(q, k, pe)

    x = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    x = rearrange(x, "B H L D -> B L (H D)")

    return x


def rope(pos: Tensor, dim: int, theta: int) -> Tensor:
    assert dim % 2 == 0
    scale = torch.arange(0, dim, 2, dtype=pos.dtype, device=pos.device) / dim
    omega = 1.0 / (theta**scale)
    out = torch.einsum("...n,d->...nd", pos, omega)
    out = torch.stack(
        [torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1
    )
    out = rearrange(out, "b n d (i j) -> b n d i j", i=2, j=2)
    return out.float()


def apply_rope(xq: Tensor, xk: Tensor, freqs_cis: Tensor) -> tuple[Tensor, Tensor]:
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
    xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
    xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
    return xq_out.reshape(*xq.shape).type_as(xq), xk_out.reshape(*xk.shape).type_as(xk)
