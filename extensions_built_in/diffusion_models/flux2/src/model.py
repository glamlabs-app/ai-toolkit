import torch
from einops import rearrange
from torch import Tensor, nn
import torch.utils.checkpoint as ckpt
import math
import os
from dataclasses import dataclass, field
from typing import Optional


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

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def enable_gradient_checkpointing(self):
        self.gradient_checkpointing = True

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

        for block in self.double_blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                img, txt = ckpt.checkpoint(
                    block,
                    img,
                    txt,
                    pe_x,
                    pe_ctx,
                    double_block_mod_img,
                    double_block_mod_txt,
                    use_reentrant=False,
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
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                img = ckpt.checkpoint(
                    block,
                    img,
                    pe,
                    single_block_mod,
                    use_reentrant=False,
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
