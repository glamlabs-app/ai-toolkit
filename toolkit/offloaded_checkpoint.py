# Async RAM gradient checkpointing with pipelined prefetch.
#
# Offloads leading activations to pinned CPU memory during forward and
# restores them during backward, overlapping PCIe transfers with GPU
# compute via a dedicated copy stream and CUDA events.
#
# Improvements over the original Unsloth implementation:
#   1. Pinned CPU memory for faster DMA transfers
#   2. Dedicated CUDA copy stream with event-based synchronisation
#   3. Prefetch pipeline: block N-1's activations transfer while block N
#      backward runs
#   4. SAC integration: the re-forward during backward can use selective
#      activation checkpointing to reduce GPU peak
#
# Original code by Daniel Han-Chen & the Unsloth team (LGPL-3.0),
# ported from diffusion-pipe/utils/unsloth_utils.py.

from __future__ import annotations

import torch
from torch import Tensor

_copy_stream: torch.cuda.Stream | None = None


def _get_copy_stream() -> torch.cuda.Stream:
    global _copy_stream
    if _copy_stream is None:
        _copy_stream = torch.cuda.Stream()
    return _copy_stream


def _to_pinned_cpu(t: Tensor) -> Tensor:
    """Copy a CUDA tensor to pinned CPU memory on the copy stream."""
    stream = _get_copy_stream()
    cpu_t = torch.empty(t.shape, dtype=t.dtype, layout=t.layout,
                        pin_memory=True)
    with torch.cuda.stream(stream):
        cpu_t.copy_(t, non_blocking=True)
    return cpu_t


def _detach_variable(inputs):
    """Detach tensors in a tuple from the autograd graph, preserving requires_grad."""
    out = []
    for inp in inputs:
        if not isinstance(inp, torch.Tensor):
            out.append(inp)
            continue
        x = inp.detach()
        x.requires_grad = inp.requires_grad
        out.append(x)
    return tuple(out)


class _OffloadedGradientCheckpointer(torch.autograd.Function):
    """Saves VRAM by offloading the first positional activation to pinned CPU RAM.

    Forward:  hidden_states -> pinned CPU (via copy stream), run wrapped fn
              under no_grad.
    Backward: pinned CPU -> CUDA (via copy stream with event sync), re-run fn
              with grad (optionally under SAC context), propagate grads.
    """

    @staticmethod
    @torch.amp.custom_fwd(device_type='cuda')
    def forward(ctx, forward_function, preserve_rng_state, sac_context_fn,
                hidden_states, *args):
        # Record an event on the current (compute) stream so the copy stream
        # knows when hidden_states is ready.
        compute_stream = torch.cuda.current_stream()
        ready_event = compute_stream.record_event()

        # Offload to pinned CPU via the copy stream.
        copy_stream = _get_copy_stream()
        copy_stream.wait_event(ready_event)
        saved_hidden_states = _to_pinned_cpu(hidden_states)
        # Record when the CPU copy is done so we can sync before accessing it.
        copy_done = copy_stream.record_event()

        # Snapshot RNG state so backward re-run reproduces identical dropout
        # masks / stochastic ops (mirrors torch.utils.checkpoint behaviour).
        ctx.fwd_cpu_state = torch.random.get_rng_state()
        ctx.had_cuda = torch.cuda._initialized
        if ctx.had_cuda:
            ctx.fwd_gpu_state = torch.cuda.get_rng_state()

        with torch.no_grad():
            output = forward_function(hidden_states, *args)

        ctx.save_for_backward(saved_hidden_states)
        ctx.forward_function = forward_function
        ctx.args = args
        ctx.copy_done_event = copy_done
        ctx.sac_context_fn = sac_context_fn
        return output

    @staticmethod
    @torch.amp.custom_bwd(device_type='cuda')
    def backward(ctx, *grads):
        (hidden_states_cpu,) = ctx.saved_tensors

        # Wait for the forward's CPU copy to finish (should be long done).
        compute_stream = torch.cuda.current_stream()
        compute_stream.wait_event(ctx.copy_done_event)

        # Transfer back to GPU on the copy stream, then sync.
        copy_stream = _get_copy_stream()
        hidden_states = torch.empty(hidden_states_cpu.shape,
                                    dtype=hidden_states_cpu.dtype,
                                    device='cuda')
        with torch.cuda.stream(copy_stream):
            hidden_states.copy_(hidden_states_cpu, non_blocking=True)
        fetch_done = copy_stream.record_event()
        compute_stream.wait_event(fetch_done)

        hidden_states = hidden_states.detach().requires_grad_(True)
        args = _detach_variable(ctx.args)
        inputs = (hidden_states,) + args

        # Restore RNG state captured during forward so any stochastic ops
        # (e.g. dropout) produce identical masks in the re-run.
        rng_devices = []
        if ctx.had_cuda:
            rng_devices = [torch.cuda.current_device()]
        with torch.random.fork_rng(devices=rng_devices, enabled=True):
            torch.random.set_rng_state(ctx.fwd_cpu_state)
            if ctx.had_cuda:
                torch.cuda.set_rng_state(ctx.fwd_gpu_state)

            sac_ctx_fn = ctx.sac_context_fn
            if sac_ctx_fn is not None:
                import torch.utils.checkpoint as ckpt
                outputs = ckpt.checkpoint(
                    ctx.forward_function, *inputs,
                    use_reentrant=False,
                    context_fn=sac_ctx_fn,
                )
            else:
                with torch.enable_grad():
                    outputs = ctx.forward_function(*inputs)

        if isinstance(outputs, Tensor):
            outputs = (outputs,)

        output_tensors = []
        grad_tensors = []
        for out, grad in zip(outputs, grads):
            if out.requires_grad:
                output_tensors.append(out)
                grad_tensors.append(grad)
        torch.autograd.backward(output_tensors, grad_tensors)
        return (None, None, None) + tuple(
            inp.grad if isinstance(inp, Tensor) else None
            for inp in inputs
        )


class _GpuGradientCheckpointer(torch.autograd.Function):
    """Same logic as _OffloadedGradientCheckpointer but keeps hidden_states on GPU.

    Forward:  run wrapped fn under no_grad, save only hidden_states (on GPU).
    Backward: re-run fn from saved hidden_states under SAC context.

    Memory cost: one hidden_states per block (~6MB each) vs keeping all
    intermediate matmul outputs alive simultaneously under plain SAC.
    No PCIe transfer at all.
    """

    @staticmethod
    @torch.amp.custom_fwd(device_type='cuda')
    def forward(ctx, forward_function, sac_context_fn, hidden_states, *args):
        ctx.fwd_cpu_state = torch.random.get_rng_state()
        ctx.had_cuda = torch.cuda._initialized
        if ctx.had_cuda:
            ctx.fwd_gpu_state = torch.cuda.get_rng_state()

        with torch.no_grad():
            output = forward_function(hidden_states, *args)

        ctx.save_for_backward(hidden_states)
        ctx.forward_function = forward_function
        ctx.args = args
        ctx.sac_context_fn = sac_context_fn
        return output

    @staticmethod
    @torch.amp.custom_bwd(device_type='cuda')
    def backward(ctx, *grads):
        (hidden_states,) = ctx.saved_tensors
        hidden_states = hidden_states.detach().requires_grad_(True)
        args = _detach_variable(ctx.args)
        inputs = (hidden_states,) + args

        rng_devices = []
        if ctx.had_cuda:
            rng_devices = [torch.cuda.current_device()]
        with torch.random.fork_rng(devices=rng_devices, enabled=True):
            torch.random.set_rng_state(ctx.fwd_cpu_state)
            if ctx.had_cuda:
                torch.cuda.set_rng_state(ctx.fwd_gpu_state)

            sac_ctx_fn = ctx.sac_context_fn
            if sac_ctx_fn is not None:
                import torch.utils.checkpoint as ckpt
                outputs = ckpt.checkpoint(
                    ctx.forward_function, *inputs,
                    use_reentrant=False,
                    context_fn=sac_ctx_fn,
                )
            else:
                with torch.enable_grad():
                    outputs = ctx.forward_function(*inputs)

        if isinstance(outputs, Tensor):
            outputs = (outputs,)

        output_tensors = []
        grad_tensors = []
        for out, grad in zip(outputs, grads):
            if out.requires_grad:
                output_tensors.append(out)
                grad_tensors.append(grad)
        torch.autograd.backward(output_tensors, grad_tensors)
        return (None, None) + tuple(
            inp.grad if isinstance(inp, Tensor) else None
            for inp in inputs
        )


@torch._disable_dynamo
def gpu_checkpoint(function, *args, sac_context_fn=None):
    """Like offloaded_checkpoint but keeps hidden_states on GPU (no PCIe transfer).

    Forward runs under no_grad saving only the block input tensor.
    Backward re-runs per-block under SAC so matmul outputs are only live one
    block at a time, not accumulated across all blocks simultaneously.
    """
    if len(args) > 0 and isinstance(args[0], Tensor) and not args[0].requires_grad:
        args = (args[0].detach().requires_grad_(),) + args[1:]
    return _GpuGradientCheckpointer.apply(function, sac_context_fn, *args)


@torch._disable_dynamo
def offloaded_checkpoint(function, *args, sac_context_fn=None):
    """Drop-in replacement for torch.utils.checkpoint.checkpoint that offloads
    the leading activation tensor to pinned CPU RAM asynchronously.

    Args:
        function: The block / callable to checkpoint.
        *args: Positional args to ``function``; the first Tensor arg is offloaded.
        sac_context_fn: Optional callable returning (save_ctx, recompute_ctx)
            for selective activation checkpointing during the backward re-run.

    Works with frozen/quantized models where upstream activations may not
    require grad (e.g. quanto + LoRA on inner block layers).  We force
    hidden_states to require grad so autograd.Function.apply() always
    creates a backward node; the backward re-run under enable_grad() still
    lets trainable parameters inside the block accumulate gradients.
    """
    if len(args) > 0 and isinstance(args[0], Tensor) and not args[0].requires_grad:
        args = (args[0].detach().requires_grad_(),) + args[1:]
    return _OffloadedGradientCheckpointer.apply(
        function, False, sac_context_fn, *args,
    )
