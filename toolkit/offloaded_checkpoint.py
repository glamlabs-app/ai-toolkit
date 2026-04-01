# Async RAM gradient checkpointing — offloads the leading activation to CPU
# during forward and restores it during backward, overlapping the PCIe
# transfer with GPU compute via non_blocking copies.
#
# Ported from diffusion-pipe/utils/unsloth_utils.py (original code by
# Daniel Han-Chen & the Unsloth team, LGPL-3.0).

import torch


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
    """Saves VRAM by offloading the first positional activation to CPU RAM.

    Forward:  hidden_states -> CPU (non_blocking), run wrapped fn under no_grad.
    Backward: CPU -> CUDA (non_blocking), re-run fn with grad, propagate grads.
    """

    @staticmethod
    @torch.amp.custom_fwd(device_type='cuda')
    def forward(ctx, forward_function, hidden_states, *args):
        saved_hidden_states = hidden_states.to('cpu', non_blocking=True)
        with torch.no_grad():
            output = forward_function(hidden_states, *args)
        ctx.save_for_backward(saved_hidden_states)
        ctx.forward_function = forward_function
        ctx.args = args
        return output

    @staticmethod
    @torch.amp.custom_bwd(device_type='cuda')
    def backward(ctx, *grads):
        (hidden_states,) = ctx.saved_tensors
        hidden_states = hidden_states.to('cuda', non_blocking=True).detach()
        hidden_states.requires_grad_(True)
        args = _detach_variable(ctx.args)
        inputs = (hidden_states,) + args
        with torch.enable_grad():
            outputs = ctx.forward_function(*inputs)

        # Handle both single-tensor and tuple returns from the wrapped function.
        if isinstance(outputs, torch.Tensor):
            outputs = (outputs,)

        output_tensors = []
        grad_tensors = []
        for out, grad in zip(outputs, grads):
            if out.requires_grad:
                output_tensors.append(out)
                grad_tensors.append(grad)
        torch.autograd.backward(output_tensors, grad_tensors)
        return (None,) + tuple(
            inp.grad if isinstance(inp, torch.Tensor) else None
            for inp in inputs
        )


@torch._disable_dynamo
def offloaded_checkpoint(function, *args):
    """Drop-in replacement for torch.utils.checkpoint.checkpoint that offloads
    the leading activation tensor to CPU RAM asynchronously.

    Works with frozen/quantized models where upstream activations may not
    require grad (e.g. quanto + LoRA on inner block layers).  We force
    hidden_states to require grad so autograd.Function.apply() always
    creates a backward node; the backward re-run under enable_grad() still
    lets trainable parameters inside the block accumulate gradients.
    """
    if len(args) > 0 and isinstance(args[0], torch.Tensor) and not args[0].requires_grad:
        args = (args[0].detach().requires_grad_(),) + args[1:]
    return _OffloadedGradientCheckpointer.apply(function, *args)
