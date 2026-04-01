from .flux2_model import Flux2Model
from transformers import Qwen3ForCausalLM, Qwen2Tokenizer
from optimum.quanto import freeze
from toolkit.util.quantize import quantize, get_qtype
from toolkit.config_modules import ModelConfig
from toolkit.memory_management.manager import MemoryManager
from toolkit.basic import flush
from .src.model import Klein9BParams, Klein4BParams


class Flux2KleinModel(Flux2Model):
    flux2_klein_te_path: str = None
    flux2_te_type: str = "qwen"  # "mistral" or "qwen"
    flux2_vae_path: str = "ai-toolkit/flux2_vae"
    flux2_is_guidance_distilled: bool = False

    def __init__(
        self,
        device,
        model_config: ModelConfig,
        dtype="bf16",
        custom_pipeline=None,
        noise_scheduler=None,
        **kwargs,
    ):
        super().__init__(
            device,
            model_config,
            dtype,
            custom_pipeline,
            noise_scheduler,
            **kwargs,
        )
        # use the new format on this new model by default
        self.use_old_lokr_format = False

    def _resolve_local_paths(self):
        """Return (te_model_path, tokenizer_path), preferring local copies."""
        import os
        if self.flux2_klein_te_path is None:
            raise ValueError("flux2_klein_te_path must be set for Flux2KleinModel")
        base = self.model_config.name_or_path
        local_te = os.path.join(base, "text_encoder")
        local_tok = os.path.join(base, "tokenizer")
        te_path = local_te if (os.path.isdir(local_te) and os.path.isfile(os.path.join(local_te, "config.json"))) else self.flux2_klein_te_path
        tok_path = local_tok if os.path.isdir(local_tok) else te_path
        return te_path, tok_path

    def load_te(self):
        te_path, tok_path = self._resolve_local_paths()
        dtype = self.torch_dtype
        te_device = self.model_config.te_device or self.device_torch
        self.print_and_status_update(f"Loading Qwen3 from {te_path}")

        text_encoder: Qwen3ForCausalLM = Qwen3ForCausalLM.from_pretrained(
            te_path,
            torch_dtype=dtype,
        )
        text_encoder.to(te_device, dtype=dtype)

        flush()

        if self.model_config.quantize_te:
            self.print_and_status_update("Quantizing Qwen3")
            quantize(text_encoder, weights=get_qtype(self.model_config.qtype))
            freeze(text_encoder)
            flush()

        if (
            self.model_config.layer_offloading
            and self.model_config.layer_offloading_text_encoder_percent > 0
        ):
            MemoryManager.attach(
                text_encoder,
                te_device,
                offload_percent=self.model_config.layer_offloading_text_encoder_percent,
            )

        tokenizer = Qwen2Tokenizer.from_pretrained(tok_path)
        return text_encoder, tokenizer


class Flux2Klein4BModel(Flux2KleinModel):
    arch = "flux2_klein_4b"
    flux2_klein_te_path: str = "Qwen/Qwen3-4B"
    flux2_te_filename: str = "flux-2-klein-base-4b.safetensors"

    def get_flux2_params(self):
        return Klein4BParams()

    def get_base_model_version(self):
        return "flux2_klein_4b"


class Flux2Klein9BModel(Flux2KleinModel):
    arch = "flux2_klein_9b"
    flux2_klein_te_path: str = "Qwen/Qwen3-8B"
    flux2_te_filename: str = "flux-2-klein-base-9b.safetensors"

    def get_flux2_params(self):
        return Klein9BParams()

    def get_base_model_version(self):
        return "flux2_klein_9b"
