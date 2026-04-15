import math
import os
from typing import TYPE_CHECKING, List, Optional

import huggingface_hub
import torch
from toolkit.config_modules import GenerateImageConfig, ModelConfig
from toolkit.memory_management.manager import MemoryManager
from toolkit.metadata import get_meta_for_safetensors
from toolkit.models.base_model import BaseModel
from toolkit.basic import flush
from toolkit.prompt_utils import PromptEmbeds
from toolkit.samplers.custom_flowmatch_sampler import (
    CustomFlowMatchEulerDiscreteScheduler,
)
from toolkit.dequantize import patch_dequantization_on_save
from toolkit.accelerator import unwrap_model
from optimum.quanto import freeze, QTensor
from toolkit.util.quantize import quantize, get_qtype, quantize_model

from transformers import AutoProcessor, Mistral3ForConditionalGeneration
from .src.model import Flux2, Flux2Params, apply_fp8_checkpoint_to_linears, apply_nvfp4_checkpoint_to_linears
from .src.pipeline import Flux2Pipeline
from .src.autoencoder import AutoEncoder, AutoEncoderParams
from safetensors.torch import load_file, save_file
from PIL import Image
import torch.nn.functional as F

if TYPE_CHECKING:
    from toolkit.data_transfer_object.data_loader import DataLoaderBatchDTO

from .src.sampling import (
    batched_prc_img,
    batched_prc_txt,
    encode_image_refs,
    scatter_ids,
)

scheduler_config = {
    "base_image_seq_len": 256,
    "base_shift": 0.5,
    "max_image_seq_len": 4096,
    "max_shift": 1.15,
    "num_train_timesteps": 1000,
    "shift": 3.0,
    "use_dynamic_shifting": True,
}

MISTRAL_PATH = "mistralai/Mistral-Small-3.1-24B-Instruct-2503"
FLUX2_VAE_FILENAME = "ae.safetensors"
FLUX2_TRANSFORMER_FILENAME = "flux2-dev.safetensors"

HF_TOKEN = os.getenv("HF_TOKEN", None)


class Flux2Model(BaseModel):
    arch = "flux2"
    flux2_te_type: str = "mistral"  # "mistral" or "qwen"
    flux2_vae_path: str = None
    flux2_te_filename: str = FLUX2_TRANSFORMER_FILENAME
    flux2_is_guidance_distilled: bool = True

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
            device, model_config, dtype, custom_pipeline, noise_scheduler, **kwargs
        )
        self.is_flow_matching = True
        self.is_transformer = True
        self.target_lora_modules = ["Flux2"]
        # control images will come in as a list for encoding some things if true
        self.has_multiple_control_images = True
        # do not resize control images
        self.use_raw_control_images = True

    # static method to get the noise scheduler
    @staticmethod
    def get_train_scheduler():
        return CustomFlowMatchEulerDiscreteScheduler(**scheduler_config)

    def get_bucket_divisibility(self):
        return 16

    def get_flux2_params(self):
        return Flux2Params()

    def load_te(self):
        dtype = self.torch_dtype
        self.print_and_status_update("Loading Mistral")

        text_encoder: Mistral3ForConditionalGeneration = (
            Mistral3ForConditionalGeneration.from_pretrained(
                MISTRAL_PATH,
                torch_dtype=dtype,
            )
        )
        text_encoder.to(self.device_torch, dtype=dtype)

        flush()

        if self.model_config.quantize_te:
            self.print_and_status_update("Quantizing Mistral")
            quantize(text_encoder, weights=get_qtype(self.model_config.qtype))
            freeze(text_encoder)
            flush()

        if (
            self.model_config.layer_offloading
            and self.model_config.layer_offloading_text_encoder_percent > 0
        ):
            MemoryManager.attach(
                text_encoder,
                self.device_torch,
                offload_percent=self.model_config.layer_offloading_text_encoder_percent,
            )

        tokenizer = AutoProcessor.from_pretrained(MISTRAL_PATH)
        return text_encoder, tokenizer

    def load_model(self):
        dtype = self.torch_dtype
        self.print_and_status_update("Loading Flux2 model")
        # will be updated if we detect a existing checkpoint in training folder
        model_path = self.model_config.name_or_path
        transformer_path = model_path

        self.print_and_status_update("Loading transformer")
        with torch.device("meta"):
            transformer = Flux2(self.get_flux2_params())

        use_fp8_native = bool(
            self.model_config.model_kwargs.get("fp8_native_inference", False)
            or getattr(self.model_config, "fp8_native_inference", False)
            or self.model_config.model_kwargs.get("fp8_native_training", False)
            or getattr(self.model_config, "fp8_native_training", False)
        )
        use_torchao_f8 = bool(
            getattr(self.model_config, "float8_torchao_training", False)
        )
        use_nvfp4_native = bool(
            self.model_config.model_kwargs.get("nvfp4_native_training", False)
            or getattr(self.model_config, "nvfp4_native_training", False)
        )
        use_fouroversix = bool(
            self.model_config.model_kwargs.get("fouroversix_fp4_training", False)
            or getattr(self.model_config, "fouroversix_fp4_training", False)
        )
        # Mutual exclusion: torchao path supersedes FP8ScaledLinear path for training
        if use_fouroversix and (use_fp8_native or use_torchao_f8 or use_nvfp4_native):
            self.print_and_status_update(
                "fouroversix_fp4_training supersedes FP8/TorchAO/NVFP4 paths"
            )
            use_fp8_native = False
            use_torchao_f8 = False
            use_nvfp4_native = False
        if use_torchao_f8 and use_fp8_native:
            self.print_and_status_update(
                "float8_torchao_training supersedes fp8_native_training; disabling FP8ScaledLinear path"
            )
            use_fp8_native = False
        if use_nvfp4_native and (use_fp8_native or use_torchao_f8):
            self.print_and_status_update(
                "nvfp4_native_training supersedes FP8/TorchAO paths"
            )
            use_fp8_native = False
            use_torchao_f8 = False

        # use local path if provided — prefer matching checkpoint files
        if use_nvfp4_native and os.path.isdir(transformer_path):
            import glob
            nvfp4_candidates = sorted(glob.glob(os.path.join(transformer_path, "*nvfp4*.safetensors")))
            if nvfp4_candidates:
                transformer_path = nvfp4_candidates[0]
                self.print_and_status_update(f"Using NVFP4 checkpoint: {os.path.basename(transformer_path)}")
        elif (use_fp8_native or use_torchao_f8) and os.path.isdir(transformer_path):
            import glob
            fp8_candidates = sorted(glob.glob(os.path.join(transformer_path, "*fp8*.safetensors")))
            if fp8_candidates:
                transformer_path = fp8_candidates[0]
                self.print_and_status_update(f"Using FP8 checkpoint: {os.path.basename(transformer_path)}")
        if os.path.isdir(transformer_path) and os.path.exists(os.path.join(transformer_path, self.flux2_te_filename)):
            transformer_path = os.path.join(transformer_path, self.flux2_te_filename)

        # Handle diffusers-format sharded weights (directory with index JSON)
        if os.path.isdir(transformer_path):
            _tdir = os.path.join(transformer_path, "transformer") if os.path.isdir(os.path.join(transformer_path, "transformer")) else transformer_path
            _index = os.path.join(_tdir, "diffusion_pytorch_model.safetensors.index.json")
            if os.path.exists(_index):
                import json as _json
                with open(_index) as _f:
                    _idx = _json.load(_f)
                _shards = sorted(set(_idx.get("weight_map", {}).values()))
                self.print_and_status_update(f"Loading {len(_shards)} sharded safetensors from {_tdir}")
                transformer_state_dict = {}
                for shard in _shards:
                    transformer_state_dict.update(load_file(os.path.join(_tdir, shard), device="cpu"))
            else:
                raise FileNotFoundError(
                    f"No single checkpoint ({self.flux2_te_filename}) or sharded index found in {transformer_path}"
                )
        else:
            if not os.path.exists(transformer_path):
                transformer_path = huggingface_hub.hf_hub_download(
                    repo_id=model_path,
                    filename=self.flux2_te_filename,
                    token=HF_TOKEN,
                )
            transformer_state_dict = load_file(transformer_path, device="cpu")
        has_fp8_weights = any(
            v.dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
            for v in transformer_state_dict.values()
        )
        has_nvfp4_weights = any(
            k.endswith(".weight_scale_2") for k in transformer_state_dict
        )

        if use_fp8_native:
            if not has_fp8_weights:
                raise RuntimeError(
                    "FP8 native mode enabled but checkpoint has no FP8 weight tensors."
                )
            if self.model_config.quantize:
                raise RuntimeError(
                    "FP8 native mode is incompatible with quantize=True; disable runtime quantization for native FP8 checkpoints."
                )
            if not hasattr(torch, "_scaled_mm"):
                raise RuntimeError(
                    "FP8 native mode requested but this torch build has no torch._scaled_mm."
                )

        if use_nvfp4_native:
            if not has_nvfp4_weights:
                raise RuntimeError(
                    "NVFP4 native mode enabled but checkpoint has no NVFP4 weight tensors "
                    "(missing weight_scale_2 keys)."
                )
            if self.model_config.quantize:
                raise RuntimeError(
                    "NVFP4 native mode is incompatible with quantize=True."
                )
            if not hasattr(torch, "_scaled_mm"):
                raise RuntimeError(
                    "NVFP4 native mode requested but this torch build has no torch._scaled_mm."
                )

        if use_torchao_f8 and self.model_config.quantize:
            raise RuntimeError(
                "float8_torchao_training is incompatible with quantize=True."
            )

        if use_nvfp4_native:
            replaced = apply_nvfp4_checkpoint_to_linears(transformer, transformer_state_dict)
            self.print_and_status_update(
                f"Installed native NVFP4 wrappers for {replaced} linear layers"
            )
        elif use_fp8_native:
            replaced = apply_fp8_checkpoint_to_linears(transformer, transformer_state_dict)
            self.print_and_status_update(
                f"Installed native FP8 wrappers for {replaced} linear layers"
            )

        # Build load dict — skip keys already handled by quantized wrappers
        load_state_dict = {}
        _SKIP_SUFFIXES = ("input_scale", "weight_scale", "weight_scale_2")
        if use_torchao_f8 and has_fp8_weights:
            scale_map = {}
            for key, value in transformer_state_dict.items():
                if key.endswith(".weight_scale"):
                    base = key[: -len(".weight_scale")]
                    scale_map[base] = value
            for key, value in transformer_state_dict.items():
                if any(key.endswith(s) for s in _SKIP_SUFFIXES):
                    continue
                if value.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
                    base = key.rsplit(".", 1)[0] if "." in key else key
                    scale = scale_map.get(base)
                    if scale is not None:
                        load_state_dict[key] = (value.to(dtype) * scale.to(dtype))
                    else:
                        load_state_dict[key] = value.to(dtype)
                else:
                    load_state_dict[key] = value.to(dtype)
            self.print_and_status_update(
                f"Dequantized FP8 checkpoint to {dtype} for TorchAO training"
            )
        elif use_nvfp4_native:
            for key, value in transformer_state_dict.items():
                if any(key.endswith(s) for s in _SKIP_SUFFIXES):
                    continue
                if value.dtype == torch.uint8:
                    continue
                if value.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
                    continue
                load_state_dict[key] = value.to(dtype)
        elif use_fp8_native:
            for key, value in transformer_state_dict.items():
                if any(key.endswith(s) for s in _SKIP_SUFFIXES):
                    continue
                if value.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
                    continue
                load_state_dict[key] = value.to(dtype)
        else:
            for key, value in transformer_state_dict.items():
                if any(key.endswith(s) for s in _SKIP_SUFFIXES):
                    continue
                if value.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
                    continue
                load_state_dict[key] = value.to(dtype)

        use_special_load = use_fp8_native or use_torchao_f8 or use_nvfp4_native
        load_result = transformer.load_state_dict(
            load_state_dict,
            strict=not use_special_load,
            assign=True,
        )
        if use_special_load and len(load_result.unexpected_keys) > 0:
            raise RuntimeError(
                f"Unexpected keys while loading model: {load_result.unexpected_keys[:10]}"
            )

        if use_fouroversix:
            transformer.to(self.device_torch, dtype=dtype)
            from .src.model import apply_fouroversix_to_linears
            replaced = apply_fouroversix_to_linears(transformer)
            self.print_and_status_update(
                f"fouroversix: quantized {replaced} linear layers to NVFP4"
            )
            flush()
        elif use_nvfp4_native:
            transformer.to(self.device_torch)
        elif use_fp8_native:
            transformer.to(self.device_torch)
        elif use_torchao_f8:
            transformer.to(self.device_torch, dtype=dtype)
            from torchao.float8 import convert_to_float8_training, Float8LinearConfig
            from dataclasses import replace
            f8_config = Float8LinearConfig.from_recipe_name("tensorwise")
            f8_config = replace(f8_config, pad_inner_dim=True)
            def _f8_filter(mod, fqn):
                if not isinstance(mod, torch.nn.Linear):
                    return False
                return mod.in_features % 16 == 0 and mod.out_features % 16 == 0
            convert_to_float8_training(transformer, config=f8_config, module_filter_fn=_f8_filter)
            n_converted = sum(1 for m in transformer.modules() if m.__class__.__name__ == "Float8Linear")
            self.print_and_status_update(
                f"TorchAO: converted {n_converted} linear layers to Float8Linear"
            )
        else:
            transformer.to(self.quantize_device, dtype=dtype)

        if self.model_config.quantize:
            # patch the state dict method
            patch_dequantization_on_save(transformer)
            self.print_and_status_update("Quantizing Transformer")
            quantize_model(self, transformer)
            flush()
        elif not use_fp8_native and not use_torchao_f8 and not use_nvfp4_native and not use_fouroversix:
            transformer.to(self.device_torch, dtype=dtype)
        flush()

        if (
            self.model_config.layer_offloading
            and self.model_config.layer_offloading_transformer_percent > 0
        ):
            MemoryManager.attach(
                transformer,
                self.device_torch,
                offload_percent=self.model_config.layer_offloading_transformer_percent,
            )

        if self.model_config.low_vram:
            self.print_and_status_update("Moving transformer to CPU")
            transformer.to("cpu")

        # TorchAO Float8Linear keeps full BF16 weights on GPU; temporarily
        # offload to CPU so the text encoder can load and be quantized.
        _torchao_offloaded = False
        if use_torchao_f8 and not self.model_config.low_vram:
            self.print_and_status_update("Offloading transformer to CPU for TE load")
            transformer.to("cpu")
            flush()
            _torchao_offloaded = True

        text_encoder, tokenizer = self.load_te()

        if _torchao_offloaded:
            self.print_and_status_update("Moving transformer back to GPU")
            transformer.to(self.device_torch)
            flush()

        self.print_and_status_update("Loading VAE")
        vae_path = self.model_config.vae_path

        if os.path.exists(os.path.join(model_path, FLUX2_VAE_FILENAME)):
            vae_path = os.path.join(model_path, FLUX2_VAE_FILENAME)

        if vae_path is None:
            vae_path = self.flux2_vae_path

        if vae_path is None or not os.path.exists(vae_path):
            p = vae_path if vae_path is not None else model_path
            # assume it is from the hub
            vae_path = huggingface_hub.hf_hub_download(
                repo_id=p,
                filename=FLUX2_VAE_FILENAME,
                token=HF_TOKEN,
            )
        with torch.device("meta"):
            vae = AutoEncoder(AutoEncoderParams())

        vae_state_dict = load_file(vae_path, device="cpu")

        # cast to dtype
        for key in vae_state_dict:
            vae_state_dict[key] = vae_state_dict[key].to(dtype)

        vae.load_state_dict(vae_state_dict, assign=True)

        self.noise_scheduler = Flux2Model.get_train_scheduler()

        self.print_and_status_update("Making pipe")

        pipe: Flux2Pipeline = Flux2Pipeline(
            scheduler=self.noise_scheduler,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            vae=vae,
            transformer=None,
            text_encoder_type=self.flux2_te_type,
            is_guidance_distilled=self.flux2_is_guidance_distilled,
        )
        # for quantization, it works best to do these after making the pipe
        pipe.transformer = transformer

        self.print_and_status_update("Preparing Model")

        text_encoder = [pipe.text_encoder]
        tokenizer = [pipe.tokenizer]

        flush()
        # just to make sure everything is on the right device and dtype
        te_device = self.model_config.te_device or self.device_torch
        text_encoder[0].to(te_device)
        text_encoder[0].requires_grad_(False)
        text_encoder[0].eval()
        pipe.transformer = pipe.transformer.to(self.device_torch)
        flush()

        # save it to the model class
        self.vae = vae
        self.text_encoder = text_encoder  # list of text encoders
        self.tokenizer = tokenizer  # list of tokenizers
        self.model = pipe.transformer
        self.pipeline = pipe
        self.print_and_status_update("Model Loaded")

    def get_generation_pipeline(self):
        scheduler = Flux2Model.get_train_scheduler()
        te_device = self.model_config.te_device or self.device_torch

        pipeline: Flux2Pipeline = Flux2Pipeline(
            scheduler=scheduler,
            text_encoder=unwrap_model(self.text_encoder[0]),
            tokenizer=self.tokenizer[0],
            vae=unwrap_model(self.vae),
            transformer=unwrap_model(self.transformer),
            text_encoder_type=self.flux2_te_type,
            is_guidance_distilled=self.flux2_is_guidance_distilled,
        )

        # Keep heavy denoiser on target GPU while optionally keeping text encoder on CPU.
        pipeline.vae.to(self.device_torch)
        pipeline.transformer.to(self.device_torch)
        pipeline.text_encoder.to(te_device)

        return pipeline

    def generate_single_image(
        self,
        pipeline: Flux2Pipeline,
        gen_config: GenerateImageConfig,
        conditional_embeds: PromptEmbeds,
        unconditional_embeds: PromptEmbeds,
        generator: torch.Generator,
        extra: dict,
    ):
        gen_config.width = (
            gen_config.width // self.get_bucket_divisibility()
        ) * self.get_bucket_divisibility()
        gen_config.height = (
            gen_config.height // self.get_bucket_divisibility()
        ) * self.get_bucket_divisibility()

        control_img_list = []
        if gen_config.ctrl_img is not None:
            control_img = Image.open(gen_config.ctrl_img)
            control_img = control_img.convert("RGB")
            control_img_list.append(control_img)
        elif gen_config.ctrl_img_1 is not None:
            control_img = Image.open(gen_config.ctrl_img_1)
            control_img = control_img.convert("RGB")
            control_img_list.append(control_img)
        if gen_config.ctrl_img_2 is not None:
            control_img = Image.open(gen_config.ctrl_img_2)
            control_img = control_img.convert("RGB")
            control_img_list.append(control_img)
        if gen_config.ctrl_img_3 is not None:
            control_img = Image.open(gen_config.ctrl_img_3)
            control_img = control_img.convert("RGB")
            control_img_list.append(control_img)

        if not self.flux2_is_guidance_distilled:
            extra["negative_prompt_embeds"] = unconditional_embeds.text_embeds

        img = pipeline(
            prompt_embeds=conditional_embeds.text_embeds,
            height=gen_config.height,
            width=gen_config.width,
            num_inference_steps=gen_config.num_inference_steps,
            guidance_scale=gen_config.guidance_scale,
            latents=gen_config.latents,
            generator=generator,
            control_img_list=control_img_list,
            **extra,
        ).images[0]
        return img

    def get_noise_prediction(
        self,
        latent_model_input: torch.Tensor,
        timestep: torch.Tensor,  # 0 to 1000 scale
        text_embeddings: PromptEmbeds,
        guidance_embedding_scale: float,
        batch: "DataLoaderBatchDTO" = None,
        **kwargs,
    ):
        with torch.no_grad():
            txt, txt_ids = batched_prc_txt(text_embeddings.text_embeds)
            packed_latents, img_ids = batched_prc_img(latent_model_input)

            # prepare image conditioning if any
            img_cond_seq: torch.Tensor | None = None
            img_cond_seq_ids: torch.Tensor | None = None

            # handle control images
            batch_control_tensor_list = batch.control_tensor_list
            if batch_control_tensor_list is None and batch.control_tensor is not None:
                batch_control_tensor_list = []
                for b in range(latent_model_input.shape[0]):
                    batch_control_tensor_list.append(batch.control_tensor[b : b + 1])

            if batch_control_tensor_list is not None:
                batch_size, num_channels_latents, height, width = (
                    latent_model_input.shape
                )

                control_image_max_res = 1024 * 1024
                if self.model_config.model_kwargs.get("match_target_res", False):
                    # use the current target size to set the control image res
                    control_image_res = (
                        height
                        * self.pipeline.vae_scale_factor
                        * width
                        * self.pipeline.vae_scale_factor
                    )
                    control_image_max_res = control_image_res

                if len(batch_control_tensor_list) != batch_size:
                    raise ValueError(
                        "Control tensor list length does not match batch size"
                    )
                for control_tensor_list in batch_control_tensor_list:
                    # control tensor list is a list of tensors for this batch item
                    controls = []
                    # pack control
                    for control_img in control_tensor_list:
                        # control images are 0 - 1 scale, shape (1, ch, height, width)
                        control_img = control_img.to(
                            self.device_torch, dtype=self.torch_dtype
                        )
                        # if it is only 3 dim, add batch dim
                        if len(control_img.shape) == 3:
                            control_img = control_img.unsqueeze(0)

                        # resize to fit within max res while keeping aspect ratio
                        if self.model_config.model_kwargs.get(
                            "match_target_res", False
                        ):
                            ratio = control_img.shape[2] / control_img.shape[3]
                            c_width = math.sqrt(control_image_res * ratio)
                            c_height = c_width / ratio

                            c_width = round(c_width / 32) * 32
                            c_height = round(c_height / 32) * 32

                            control_img = F.interpolate(
                                control_img, size=(c_height, c_width), mode="bilinear"
                            )

                        # scale to -1 to 1
                        control_img = control_img * 2 - 1
                        controls.append(control_img)

                    img_cond_seq_item, img_cond_seq_ids_item = encode_image_refs(
                        self.vae, controls, limit_pixels=control_image_max_res
                    )
                    if img_cond_seq is None:
                        img_cond_seq = img_cond_seq_item
                        img_cond_seq_ids = img_cond_seq_ids_item
                    else:
                        img_cond_seq = torch.cat(
                            (img_cond_seq, img_cond_seq_item), dim=0
                        )
                        img_cond_seq_ids = torch.cat(
                            (img_cond_seq_ids, img_cond_seq_ids_item), dim=0
                        )

            img_input = packed_latents
            img_input_ids = img_ids

            if img_cond_seq is not None:
                assert img_cond_seq_ids is not None, (
                    "You need to provide either both or neither of the sequence conditioning"
                )
                img_input = torch.cat((img_input, img_cond_seq), dim=1)
                img_input_ids = torch.cat((img_input_ids, img_cond_seq_ids), dim=1)

            guidance_vec = torch.full(
                (img_input.shape[0],),
                guidance_embedding_scale,
                device=img_input.device,
                dtype=img_input.dtype,
            )

            cast_dtype = self.model.dtype

        packed_noise_pred = self.transformer(
            x=img_input.to(self.device_torch, cast_dtype),
            x_ids=img_input_ids.to(self.device_torch),
            timesteps=timestep.to(self.device_torch, cast_dtype) / 1000,
            ctx=txt.to(self.device_torch, cast_dtype),
            ctx_ids=txt_ids.to(self.device_torch),
            guidance=guidance_vec.to(self.device_torch, cast_dtype),
        )

        if img_cond_seq is not None:
            packed_noise_pred = packed_noise_pred[:, : packed_latents.shape[1]]

        if isinstance(packed_noise_pred, QTensor):
            packed_noise_pred = packed_noise_pred.dequantize()

        noise_pred = torch.cat(scatter_ids(packed_noise_pred, img_ids)).squeeze(2)

        return noise_pred

    def get_prompt_embeds(self, prompt: str) -> PromptEmbeds:
        te_device = self.model_config.te_device or self.device_torch
        if self.pipeline.text_encoder.device != te_device:
            self.pipeline.text_encoder.to(te_device)

        prompt_embeds, prompt_embeds_mask = self.pipeline.encode_prompt(
            prompt, device=te_device
        )
        pe = PromptEmbeds(prompt_embeds.to(self.device_torch))
        return pe

    def get_model_has_grad(self):
        return False

    def get_te_has_grad(self):
        return False

    def save_model(self, output_path, meta, save_dtype):
        if not output_path.endswith(".safetensors"):
            output_path = output_path + ".safetensors"
        # only save the unet
        transformer: Flux2 = unwrap_model(self.model)
        state_dict = transformer.state_dict()
        save_dict = {}
        for k, v in state_dict.items():
            if isinstance(v, QTensor):
                v = v.dequantize()
            save_dict[k] = v.clone().to("cpu", dtype=save_dtype)

        meta = get_meta_for_safetensors(meta, name="flux2")
        save_file(save_dict, output_path, metadata=meta)

    def get_loss_target(self, *args, **kwargs):
        noise = kwargs.get("noise")
        batch = kwargs.get("batch")
        return (noise - batch.latents).detach()

    def get_base_model_version(self):
        return "flux2"

    def get_transformer_block_names(self) -> Optional[List[str]]:
        return ["double_blocks", "single_blocks"]

    def convert_lora_weights_before_save(self, state_dict):
        new_sd = {}
        for key, value in state_dict.items():
            new_key = key.replace("transformer.", "diffusion_model.")
            new_key = new_key.replace("._orig_mod.", ".")
            new_key = new_key.replace("_orig_mod.", "")
            new_sd[new_key] = value
        return new_sd

    def convert_lora_weights_before_load(self, state_dict):
        new_sd = {}
        for key, value in state_dict.items():
            new_key = key.replace("diffusion_model.", "transformer.")
            new_sd[new_key] = value
        return new_sd

    def encode_images(self, image_list: List[torch.Tensor], device=None, dtype=None):
        if device is None:
            device = self.vae_device_torch
        if dtype is None:
            dtype = self.vae_torch_dtype

        # Move to vae to device if on cpu
        if self.vae.device == torch.device("cpu"):
            self.vae.to(device)
        # move to device and dtype
        image_list = [image.to(device, dtype=dtype) for image in image_list]
        images = torch.stack(image_list).to(device, dtype=dtype)

        latents = self.vae.encode(images)

        return latents
