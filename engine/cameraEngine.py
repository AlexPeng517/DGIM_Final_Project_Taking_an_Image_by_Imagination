from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List, Union, Tuple, Literal

import torch
import torch.nn.functional as F
from PIL import Image

from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, StableDiffusionControlNetImg2ImgPipeline
from diffusers.schedulers import PNDMScheduler

from camera_embed import CameraSettingEmbedding
from inference import embed_camera_settings
from tqdm import tqdm

from transformers import (
    CLIPImageProcessor,
    CLIPVisionModelWithProjection,
    AutoImageProcessor,
    AutoModel,
)
import matplotlib.pyplot as plt


FeatureType = Literal["clip", "dino"]


@dataclass
class CameraSettings:
    focal_length: torch.float16 = 50.0
    f_number: torch.float16 = 4.0
    iso_speed_rating: torch.float16 = 100.0
    exposure_time: torch.float16 = 0.01


class CameraAlignedEditingEngine:
    """
    Diffusion editing engine for SD2.1:
      - Img2Img init image for fidelity
      - ControlNet (depth) for structure (optional)
      - Camera-settings embedding + LoRA for camera-aware rendering
      - IP-Adapter for semantic anchoring (as confirmed compatible in your env)

    Public API:
      - run_simulation(...)
      - run_variation(...)
      - __call__(...)  # single-shot (same as run_simulation with explicit knobs)
    """

    def __init__(
        self,
        *,
        model_id: str = "Manojb/stable-diffusion-2-1-base",
        controlnet_model_id: str = "thibaud/controlnet-sd21-depth-diffusers",
        camera_setting_embedding_id: str = "ishengfang/Camera-Settings-as-Tokens-SD2",
        device: str = "cuda:1",
        torch_dtype: Optional[torch.dtype] = torch.float32,
        use_safetensors: bool = True,
    ):
        self.device = device
        self.use_safetensors = use_safetensors

        self.torch_dtype = torch_dtype

        self.pipe: Optional[StableDiffusionControlNetPipeline] = None
        self.cam_embed: Optional[CameraSettingEmbedding] = None

        self._load_pipeline(
            model_id=model_id,
            controlnet_model_id=controlnet_model_id,
            camera_setting_embedding_id=camera_setting_embedding_id,
        )

        # Sensible defaults (override via set_scales / per-call)
        self.set_scales(lora_scale=0.55, ip_adapter_scale=0.6)

    def _load_pipeline(
        self,
        *,
        model_id: str,
        controlnet_model_id: str,
        camera_setting_embedding_id: str,
    ) -> None:
        # ControlNet + Text2Image pipeline
        controlnet = ControlNetModel.from_pretrained(controlnet_model_id, torch_dtype=self.torch_dtype)
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            model_id,
            controlnet=controlnet,
            torch_dtype=self.torch_dtype,
            use_safetensors=self.use_safetensors,
        )
        self.pipe.scheduler = PNDMScheduler.from_pretrained(model_id, subfolder="scheduler")

        # Camera token embedding module + LoRA
        self.cam_embed = CameraSettingEmbedding.from_pretrained(
            camera_setting_embedding_id, subfolder="cam_embed"
        )
        self.pipe.load_lora_weights(camera_setting_embedding_id, adapter_name="camera-settings-as-tokens")


        # Move to device
        self.pipe.to(self.device)
        self.cam_embed.to(self.device)

    # ----------------------------
    # Utility helpers
    # ----------------------------
    @staticmethod
    def _rgb(img: Image.Image) -> Image.Image:
        return img.convert("RGB") if img.mode != "RGB" else img

    @staticmethod
    def _resize_like(img: Image.Image, ref: Image.Image) -> Image.Image:
        return img if img.size == ref.size else img.resize(ref.size, resample=Image.BICUBIC)

    def _make_depth(self, src: Image.Image) -> Image.Image:
        """
        Compute depth condition via MiDaS (controlnet_aux).
        Install: pip install controlnet-aux
        """
        from controlnet_aux import MidasDetector

        midas = MidasDetector.from_pretrained("lllyasviel/ControlNet")
        return midas(self._rgb(src))

    def _make_canny(self, src: Image.Image) -> Image.Image:
        """
        Compute canny edge condition via HED (controlnet_aux).
        Install: pip install controlnet-aux
        """
        from controlnet_aux import HEDdetector

        hed = HEDdetector.from_pretrained("lllyasviel/Annotators")
        return hed(self._rgb(src))

    def set_scales(self, *, lora_scale: float, ip_adapter_scale: float) -> None:
        """
        Persistently set adapter scales for subsequent calls.
        """
        assert self.pipe is not None, "Pipeline not loaded."
        self.pipe.set_adapters(["camera-settings-as-tokens"], adapter_weights=[float(lora_scale)])
        self.pipe.set_ip_adapter_scale(float(ip_adapter_scale))

    # ----------------------------
    # Core single-shot call
    # ----------------------------
    @torch.no_grad()
    def __call__(
        self,
        *,
        original_image: Image.Image,
        camera: CameraSettings,
        prompt: str,
        negative_prompt: str,
        reference_image: Optional[Image.Image] = None,
        control_image: Optional[Image.Image] = None,
        strength: float = 0.35,
        controlnet_scale: float = 0.8,
        num_inference_steps: int = 40,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None,
        lora_scale: Optional[float] = None,
        ip_adapter_scale: Optional[float] = None,
    ) -> Image.Image:
        """
        Produce one camera-aligned, semantic-preserving edited image.
        """
        assert self.pipe is not None and self.cam_embed is not None, "Engine not initialized."

        original_image = self._rgb(original_image)

        if reference_image is None:
            reference_image = original_image
        else:
            reference_image = self._resize_like(self._rgb(reference_image), original_image)

        if control_image is None:
            control_image = self._make_depth(original_image)
        
        control_image = self._resize_like(control_image, original_image)


        if (lora_scale is not None) or (ip_adapter_scale is not None):
            self.set_scales(
                lora_scale=float(0.55 if lora_scale is None else lora_scale),
                ip_adapter_scale=float(0.6 if ip_adapter_scale is None else ip_adapter_scale),
            )

        generator = torch.Generator(device=self.pipe._execution_device)
        if seed is not None:
            generator.manual_seed(int(seed))

        # Encode prompt and inject camera settings
        prompt_embeds, negative_prompt_embeds = self.pipe.encode_prompt(
            prompt=prompt,
            device=self.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=True,
            negative_prompt=negative_prompt,
        )

        prompt_embeds, negative_prompt_embeds = embed_camera_settings(
            camera.focal_length,
            camera.f_number,
            camera.iso_speed_rating,
            camera.exposure_time,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            cam_embed=self.cam_embed,
            device=self.device,
        )

        # Note: StableDiffusionControlNetPipeline (Text2Image) uses 'image' argument for the ControlNet input.
        # It does NOT take an init image for img2img.
        # 'strength' is ignored in standard T2I ControlNet.
        out = self.pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            image=control_image,
            controlnet_conditioning_scale=float(controlnet_scale),
            guidance_scale=float(guidance_scale),
            num_inference_steps=int(num_inference_steps),
            generator=generator,
        )
        return out.images[0]

    # ----------------------------
    # Draft-style API you requested
    # ----------------------------
    def run_simulation(
        self,
        *,
        source_image: Image.Image,
        camera: CameraSettings,
        prompt: str,
        negative_prompt: str = (
            "ugly, duplicate, morbid, mutilated, out of frame, extra fingers, mutated hands, "
            "poorly drawn hands, poorly drawn face, mutation, deformed, blurry, bad anatomy, "
            "bad proportions, extra limbs, cloned face, disfigured, malformed limbs, missing arms, "
            "missing legs, too many fingers, long neck"
        ),
        strength: float = 0.35,
        controlnet_scale: float = 0.8,
        ip_adapter_scale: float = 0.6,
        lora_scale: float = 0.55,
        num_inference_steps: int = 40,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None,
        # Optional explicit conditioning:
        control_image: Optional[Image.Image] = None,
        reference_image: Optional[Image.Image] = None,
    ) -> Image.Image:
        """
        "Simulation" = faithful camera-aligned rendering under new camera settings.

        Recommended defaults:
          - strength ~ 0.25-0.45 for faithfulness
          - ip_adapter_scale ~ 0.6-0.9 for semantic preservation
          - controlnet_scale ~ 0.6-1.0 for structure adherence
        """
        return self(
            original_image=source_image,
            camera=camera,
            prompt=prompt,
            negative_prompt=negative_prompt,
            reference_image=reference_image,
            control_image=control_image,
            strength=strength,
            controlnet_scale=controlnet_scale,
            ip_adapter_scale=ip_adapter_scale,
            lora_scale=lora_scale,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            seed=seed,
        )

    def run_variation(
        self,
        *,
        source_image: Image.Image,
        camera: CameraSettings,
        prompt: str,
        negative_prompt: str = (
            "ugly, duplicate, morbid, mutilated, out of frame, extra fingers, mutated hands, "
            "poorly drawn hands, poorly drawn face, mutation, deformed, blurry, bad anatomy, "
            "bad proportions, extra limbs, cloned face, disfigured, malformed limbs, missing arms, "
            "missing legs, too many fingers, long neck"
        ),
        # Variation-specific knobs:
        num_variations: int = 4,
        seeds: Optional[Union[List[int], Tuple[int, ...]]] = None,
        strength: float = 0.55,
        controlnet_scale: float = 0.75,
        ip_adapter_scale: float = 0.55,
        lora_scale: float = 0.55,
        num_inference_steps: int = 40,
        guidance_scale: float = 7.5,
        # Optional explicit conditioning:
        control_image: Optional[Image.Image] = None,
        reference_image: Optional[Image.Image] = None,
    ) -> List[Image.Image]:
        """
        "Variation" = produce multiple plausible camera-aligned variants.

        Default differences vs simulation:
          - higher strength to allow more diversity
          - slightly lower ip_adapter_scale to avoid over-copying
          - multiple seeds (fresh noise) for diversity

        Seeds:
          - If seeds is None, random seeds are sampled.
          - If seeds provided, len(seeds) should be >= num_variations.
        """
        if seeds is None:
            # Use Python's RNG without importing random globally
            import random
            seeds = [random.randint(0, 2**31 - 1) for _ in range(num_variations)]
        else:
            seeds = list(seeds)
            if len(seeds) < num_variations:
                raise ValueError(f"len(seeds)={len(seeds)} < num_variations={num_variations}")

        outputs: List[Image.Image] = []
        for i in range(num_variations):
            img = self(
                original_image=source_image,
                camera=camera,
                prompt=prompt,
                negative_prompt=negative_prompt,
                reference_image=reference_image,
                control_image=control_image,
                strength=strength,
                controlnet_scale=controlnet_scale,
                ip_adapter_scale=ip_adapter_scale,
                lora_scale=lora_scale,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                seed=int(seeds[i]),
            )
            outputs.append(img)
        return outputs


class PnPFeatureGuidedEditor:
    """
    SD2.1 Img2Img (+ optional ControlNet) + camera-token LoRA,
    with denoising-time gradient guidance using ONE backbone:
      - CLIP image_embeds OR DINOv2 CLS embedding

    IMPORTANT:
      - Feature backbone is on GPU (no CPU offload) to keep gradients valid.
      - Feature loss is computed from torch tensors (no PIL/numpy), preserving autograd graph.
    """

    def __init__(
        self,
        *,
        model_id: str = "Manojb/stable-diffusion-2-1-base",
        controlnet_model_id: Optional[str] = "thibaud/controlnet-sd21-depth-diffusers",
        camera_setting_embedding_id: str = "ishengfang/Camera-Settings-as-Tokens-SD2",
        device: str = "cuda:1",
        torch_dtype: torch.dtype = torch.float16,
        use_safetensors: bool = True,
        # Choose ONE backbone:
        feature_type: FeatureType = "clip",
        # Backbone IDs
        clip_vision_id: str = "openai/clip-vit-large-patch14",
        dino_vision_id: str = "facebook/dinov2-base",
        # feature resolution (224 is standard; 196 or 160 can reduce compute)
        feature_res: int = 224,
    ):
        self.device = device
        self.torch_dtype = torch_dtype
        self.feature_res = int(feature_res)

        # ---- Diffusion pipeline (ControlNet optional)
        controlnet = None
        if controlnet_model_id is not None:
            controlnet = ControlNetModel.from_pretrained(controlnet_model_id, torch_dtype=torch_dtype)

        self.pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
            model_id,
            controlnet=controlnet,
            torch_dtype=torch_dtype,
            use_safetensors=use_safetensors,
        )
        self.pipe.scheduler = PNDMScheduler.from_pretrained(model_id, subfolder="scheduler")

        # Camera embedding + LoRA
        self.cam_embed = CameraSettingEmbedding.from_pretrained(
            camera_setting_embedding_id, subfolder="cam_embed"
        )
        self.pipe.load_lora_weights(camera_setting_embedding_id, adapter_name="camera-settings-as-tokens")

        self.pipe.to(device)
        self.pipe.enable_attention_slicing()
        self.pipe.enable_xformers_memory_efficient_attention()

        self.cam_embed = self.cam_embed.to(self.device, dtype=self.torch_dtype)


        # ---- Load ONE feature backbone
        self.clip_vision_id = clip_vision_id
        self.dino_vision_id = dino_vision_id

        self.feature_type: Optional[FeatureType] = None
        self.feature_proc = None
        self.feature_model = None

        self.set_feature_backbone(feature_type)

        # Defaults
        self.set_scales(lora_scale=0.55)

    # --------------------------
    # Backbone management (GPU only)
    # --------------------------
    def set_feature_backbone(self, feature: FeatureType) -> None:
        """
        Load exactly one backbone on GPU. Frees previous one (and clears CUDA cache).
        """
        if self.feature_type == feature and self.feature_model is not None:
            return

        # Free old
        if self.feature_model is not None:
            del self.feature_model
            self.feature_model = None
        self.feature_proc = None
        self.feature_type = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Load requested
        if feature == "clip":
            self.feature_proc = CLIPImageProcessor.from_pretrained(self.clip_vision_id)
            self.feature_model = CLIPVisionModelWithProjection.from_pretrained(
                self.clip_vision_id,
                torch_dtype=torch.float32,  # keep feature math stable
                low_cpu_mem_usage=True,
                use_safetensors=True
            ).to(self.device).eval()
        elif feature == "dino":
            self.feature_proc = AutoImageProcessor.from_pretrained(self.dino_vision_id)
            self.feature_model = AutoModel.from_pretrained(
                self.dino_vision_id,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
                use_safetensors=True
            ).to(self.device).eval()
        else:
            raise ValueError(f"Unknown feature backbone: {feature}")

        self.feature_type = feature

        # Freeze params (still allows grad w.r.t. inputs)
        for p in self.feature_model.parameters():
            p.requires_grad_(False)

    # --------------------------
    # Diffusion config
    # --------------------------
    def set_scales(self, *, lora_scale: float) -> None:
        self.pipe.set_adapters(["camera-settings-as-tokens"], adapter_weights=[float(lora_scale)])

    @staticmethod
    def _rgb(img: Image.Image) -> Image.Image:
        return img.convert("RGB") if img.mode != "RGB" else img

    @staticmethod
    def _resize_like(img: Image.Image, ref: Image.Image) -> Image.Image:
        return img if img.size == ref.size else img.resize(ref.size, resample=Image.BICUBIC)

    def _make_depth(self, src: Image.Image) -> Image.Image:
        from controlnet_aux import MidasDetector
        midas = MidasDetector.from_pretrained("lllyasviel/ControlNet")
        return midas(self._rgb(src))

    def _pil_to_img01(self, img: Image.Image) -> torch.Tensor:
        """
        PIL RGB -> torch float tensor [1,3,H,W] in [0,1] on self.device.
        No numpy() detours, no graph needed for reference.
        """
        img = self._rgb(img)
        # use diffusers preprocess then map to [0,1]
        t = self.pipe.image_processor.preprocess(img).to(self.device, dtype=torch.float32)  # [1,3,H,W] in [-1,1]
        return (t / 2 + 0.5).clamp(0, 1)

    # --------------------------
    # Differentiable feature forward (torch tensor only)
    # --------------------------
    def _feat_from_img01(self, img01: torch.Tensor) -> torch.Tensor:
        """
        img01: [B,3,H,W] float in [0,1] on GPU. May require_grad.
        returns: [B,D] normalized, keeping autograd path to img01 (and thus to latents).
        """
        assert self.feature_model is not None and self.feature_proc is not None
        assert self.feature_type in ("clip", "dino")

        x = F.interpolate(
            img01, size=(self.feature_res, self.feature_res),
            mode="bilinear", align_corners=False
        )

        # Normalize using processor mean/std
        mean = torch.tensor(self.feature_proc.image_mean, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        std = torch.tensor(self.feature_proc.image_std, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        x = (x - mean) / std

        if self.feature_type == "clip":
            emb = self.feature_model(pixel_values=x).image_embeds  # [B,D]
            return F.normalize(emb, dim=-1)
        else:
            out = self.feature_model(pixel_values=x)
            cls = out.last_hidden_state[:, 0, :]
            return F.normalize(cls, dim=-1)

    # --------------------------
    # Main edit with PnP guidance
    # --------------------------
    def run_simulation(
        self,
        *,
        source_image: Image.Image,
        camera: CameraSettings,
        prompt: str,
        negative_prompt: str = (
            "ugly, duplicate, morbid, mutilated, out of frame, extra fingers, mutated hands, "
            "poorly drawn hands, poorly drawn face, mutation, deformed, blurry, bad anatomy, "
            "bad proportions, extra limbs, cloned face, disfigured, malformed limbs, missing arms, "
            "missing legs, extra legs, long neck"
        ),
        strength: float = 0.42,
        controlnet_scale: float = 0.8,
        lora_scale: float = 0.55,
        seed: Optional[int] = None,
        num_inference_steps: int = 40,
        guidance_scale: float = 7.5,
        control_image: Optional[Image.Image] = None,
        reference_image: Optional[Image.Image] = None,
        # Feature guidance knobs
        feat_weight: float = 2.0,
        feat_step_size: float = 0.10,
        feat_start_frac: float = 0.05,
        feat_end_frac: float = 0.55,
        feat_decay: Literal["linear", "cosine", "constant"] = "linear",
        feat_every_k: int = 5,
    ) -> Image.Image:
        """
        Same API style as your engine calls.
        """
        self.set_scales(lora_scale=lora_scale)

        source_image = self._rgb(source_image)
        if reference_image is None:
            reference_image = source_image
        else:
            reference_image = self._resize_like(self._rgb(reference_image), source_image)

        if control_image is None and self.pipe.controlnet is not None:
            control_image = self._make_depth(source_image)
        if control_image is not None:
            control_image = self._resize_like(control_image, source_image)

        # Precompute reference feature (no grad required)
        with torch.no_grad():
            ref_img01 = self._pil_to_img01(reference_image)  # [1,3,H,W] in [0,1]
            ref_feat = self._feat_from_img01(ref_img01)      # [1,D]

        generator = torch.Generator(device=self.pipe._execution_device)
        if seed is not None:
            generator.manual_seed(int(seed))

        # Encode prompt and inject camera settings
        with torch.no_grad():
            prompt_embeds, negative_prompt_embeds = self.pipe.encode_prompt(
                prompt=prompt,
                device=self.device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
            )
            prompt_embeds, negative_prompt_embeds = embed_camera_settings(
                camera.focal_length,
                camera.f_number,
                camera.iso_speed_rating,
                camera.exposure_time,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                cam_embed=self.cam_embed,
                device=self.device,
            )

        # Timesteps / strength
        self.pipe.scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps = self.pipe.scheduler.timesteps

        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)
        t_start = max(num_inference_steps - init_timestep, 0)
        timesteps = timesteps[t_start:]

        # Init image -> latents
        init_image_tensor = self.pipe.image_processor.preprocess(source_image).to(
            self.device, dtype=self.torch_dtype
        )
        latents = self.pipe.prepare_latents(
            init_image_tensor,
            timestep=timesteps[0],
            batch_size=1,
            num_images_per_prompt=1,
            dtype=self.torch_dtype,
            device=self.device,
            generator=generator,
        )

        # Control tensor
        if control_image is not None:
            control_tensor = self.pipe.image_processor.preprocess(control_image).to(
                self.device, dtype=self.torch_dtype
            )
        else:
            control_tensor = None

        # Feature-guidance window
        n = len(timesteps)
        s0 = max(0, min(int(feat_start_frac * n), n - 1))
        s1 = max(s0 + 1, min(int(feat_end_frac * n), n))

        def weight_at(i: int) -> float:
            if i < s0 or i >= s1:
                return 0.0
            u = (i - s0) / max(1, (s1 - s0 - 1))
            if feat_decay == "linear":
                return float(feat_weight) * (1.0 - u)
            if feat_decay == "cosine":
                return float(feat_weight) * (0.5 * (1.0 + torch.cos(torch.tensor(u * 3.1415926535))).item())
            return float(feat_weight)

        # Denoise
        for i, t in tqdm(enumerate(timesteps), total=len(timesteps), desc="Denoising"):

            # -----------------------------
            # 1) Normal denoising step: NO GRAD
            # -----------------------------
            with torch.no_grad():
                latent_in = torch.cat([latents] * 2)
                latent_in = self.pipe.scheduler.scale_model_input(latent_in, t)
                enc = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)

                down_samples = mid_sample = None
                if self.pipe.controlnet is not None and control_tensor is not None:
                    ctrl_in = torch.cat([control_tensor] * 2)
                    down_samples, mid_sample = self.pipe.controlnet(
                        latent_in,
                        t,
                        encoder_hidden_states=enc,
                        controlnet_cond=ctrl_in,
                        conditioning_scale=float(controlnet_scale),
                        return_dict=False,
                    )

                noise_pred = self.pipe.unet(
                    latent_in,
                    t,
                    encoder_hidden_states=enc,
                    down_block_additional_residuals=down_samples,
                    mid_block_additional_residual=mid_sample,
                    return_dict=False,
                )[0]

                n_uncond, n_text = noise_pred.chunk(2)
                noise_pred = n_uncond + float(guidance_scale) * (n_text - n_uncond)

                latents = self.pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

            # -----------------------------
            # 2) PnP feature step: GRAD ONLY HERE
            # -----------------------------
            w = weight_at(i)
            if w > 0.0 and feat_step_size > 0.0 and (i % max(1, int(feat_every_k)) == 0):
                with torch.enable_grad():
                    latents_g = latents.detach().requires_grad_(True)

                    scaled = self.pipe.scheduler.scale_model_input(latents_g, t)
                    noise_c = self.pipe.unet(
                        scaled,
                        t,
                        encoder_hidden_states=prompt_embeds,
                        return_dict=False,
                    )[0]

                    step_out = self.pipe.scheduler.step(noise_c, t, latents_g, return_dict=True)
                    x0 = step_out.pred_original_sample if getattr(step_out, "pred_original_sample", None) is not None else latents_g

                    x0_vae = x0 / self.pipe.vae.config.scaling_factor
                    img = self.pipe.vae.decode(x0_vae, return_dict=False)[0]  # [-1,1]
                    img01 = (img / 2 + 0.5).clamp(0, 1)

                    feat = self._feat_from_img01(img01)  # must be differentiable torch path
                    loss = (1.0 - (feat * ref_feat).sum(dim=-1)).mean() * float(w)

                    grad = torch.autograd.grad(loss, latents_g, retain_graph=False, create_graph=False)[0]
                    grad = grad / (grad.norm(p=2) + 1e-8)

                    latents = (latents_g - float(feat_step_size) * grad).detach()

                    # (optional) help GC drop refs ASAP
                    del latents_g, scaled, noise_c, step_out, x0, x0_vae, img, img01, feat, loss, grad

        # Decode final
        with torch.no_grad():
            z = latents / self.pipe.vae.config.scaling_factor
            img = self.pipe.vae.decode(z, return_dict=False)[0]
            img01 = (img / 2 + 0.5).clamp(0, 1)

        arr = img01[0].permute(1, 2, 0).cpu().numpy()
        return Image.fromarray((arr * 255).astype("uint8"))



FeatureType = Literal["clip", "dino"]
SemanticFeatureMode = Literal["clip_global", "dino_cls"]


class EnhancedPnPFeatureGuidedEditor(PnPFeatureGuidedEditor):
    """
    Enhanced editor on top of PnPFeatureGuidedEditor with:
      1) Auto reference clean-up pass (optional) to avoid preserving "bad photo" artifacts.
      2) Coarse semantic preservation feature guidance (default: CLIP global embedding)
         to avoid retaining too many low-level bad details.
      3) LoRA scheduling (camera effect stronger late).
      4) ControlNet scheduling (structure lock strong early, relaxed late).

    Explicitly NOT included:
      - gradient-domain structure loss
      - self-attention injection (P2P)
      - DINO patch-token loss (intentionally avoided here due to "bad photo" retention)
    """

    # --------------------------
    # Scheduling helpers
    # --------------------------
    @staticmethod
    def _lerp(a: float, b: float, u: float) -> float:
        u = float(max(0.0, min(1.0, u)))
        return float(a + (b - a) * u)

    def _schedule_multiplier(
        self,
        frac: float,
        *,
        start_frac: float,
        end_frac: float,
        start_mult: float,
        end_mult: float,
    ) -> float:
        """
        Piecewise linear multiplier over progress fraction frac in [0,1].
        - before start_frac: start_mult
        - after  end_frac:   end_mult
        - linear in between
        """
        if frac <= start_frac:
            return float(start_mult)
        if frac >= end_frac:
            return float(end_mult)
        u = (frac - start_frac) / max(1e-8, (end_frac - start_frac))
        return self._lerp(start_mult, end_mult, u)

    # --------------------------
    # Coarse semantic features (avoid preserving bad low-level details)
    # --------------------------
    def _feat_semantic_from_img01(self, img01: torch.Tensor, mode: SemanticFeatureMode) -> torch.Tensor:
        """
        img01: [B,3,H,W] in [0,1] on GPU; may require_grad.
        Returns:
          - clip_global: [B, D] normalized (CLIP image_embeds)
          - dino_cls:    [B, D] normalized (DINO CLS)
        """
        x = F.interpolate(
            img01, size=(self.feature_res, self.feature_res),
            mode="bilinear", align_corners=False
        )

        mean = torch.tensor(self.feature_proc.image_mean, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        std = torch.tensor(self.feature_proc.image_std, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        x = (x - mean) / std

        if mode == "clip_global":
            # Ensure we are using CLIP backbone for this mode
            if self.feature_type != "clip":
                raise RuntimeError("clip_global mode requires feature backbone = 'clip'. Call set_feature_backbone('clip').")
            emb = self.feature_model(pixel_values=x).image_embeds  # [B,D]
            return F.normalize(emb, dim=-1)

        if mode == "dino_cls":
            # Ensure we are using DINO backbone for this mode
            if self.feature_type != "dino":
                raise RuntimeError("dino_cls mode requires feature backbone = 'dino'. Call set_feature_backbone('dino').")
            out = self.feature_model(pixel_values=x)
            cls = out.last_hidden_state[:, 0, :]  # [B,D]
            return F.normalize(cls, dim=-1)

        raise ValueError(f"Unknown semantic feature mode: {mode}")

    @staticmethod
    def _cosine_dist_loss(feat: torch.Tensor, ref_feat: torch.Tensor) -> torch.Tensor:
        """
        Both [B,D] normalized. Returns scalar loss.
        """
        return (1.0 - (feat * ref_feat).sum(dim=-1)).mean()

    # --------------------------
    # Auto reference clean-up pass
    # --------------------------
    @torch.no_grad()
    def _auto_cleanup_reference(
        self,
        *,
        source_image: Image.Image,
        prompt: str,
        negative_prompt: str,
        control_image: Optional[Image.Image],
        cleanup_strength: float,
        cleanup_steps: int,
        cleanup_guidance_scale: float,
        cleanup_lora_scale: float,
        cleanup_controlnet_scale: float,
        seed: Optional[int],
        # camera token handling: keep it neutral by default
        use_camera_tokens: bool,
        camera: Optional["CameraSettings"],
        semantic_mode: SemanticFeatureMode,
    ) -> Image.Image:
        """
        Produces a "cleaner" reference image that:
          - preserves scene structure (ControlNet optional)
          - reduces bad photometric artifacts (via prompt and mild denoising)
          - minimizes camera-token influence (cleanup_lora_scale default 0.0)
        """
        # Temporarily set LoRA scale for cleanup (often 0.0)
        self.set_scales(lora_scale=float(cleanup_lora_scale))

        # Prepare generator
        generator = torch.Generator(device=self.pipe._execution_device)
        if seed is not None:
            generator.manual_seed(int(seed) + 1000003)  # decorrelate from main run slightly

        # Encode prompt (+ optional camera tokens, but usually off)
        prompt_embeds, negative_prompt_embeds = self.pipe.encode_prompt(
            prompt=prompt,
            device=self.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=True,
            negative_prompt=negative_prompt,
        )

        if use_camera_tokens:
            if camera is None:
                raise ValueError("use_camera_tokens=True requires camera to be provided.")
            prompt_embeds, negative_prompt_embeds = embed_camera_settings(
                camera.focal_length,
                camera.f_number,
                camera.iso_speed_rating,
                camera.exposure_time,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                cam_embed=self.cam_embed,
                device=self.device,
            )

        # If ControlNet exists and control_image provided, pass it; otherwise None.
        # For Img2Img pipeline call, Diffusers uses parameter name "control_image" for some pipelines.
        # For StableDiffusionControlNetImg2ImgPipeline, it's "control_image".
        kwargs = {}
        if self.pipe.controlnet is not None and control_image is not None:
            kwargs["control_image"] = control_image
            kwargs["controlnet_conditioning_scale"] = float(cleanup_controlnet_scale)

        out = self.pipe(
            image=source_image,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            strength=float(cleanup_strength),
            num_inference_steps=int(cleanup_steps),
            guidance_scale=float(cleanup_guidance_scale),
            generator=generator,
            **kwargs,
        ).images[0]

        # Restore: main run will set scheduled LoRA anyway; no need to reset here.
        return out

    # --------------------------
    # Plotting helper
    # --------------------------
    def _plot_schedules(
        self,
        num_steps: int,
        lora_schedule: Tuple[float, float, float, float],
        controlnet_schedule: Tuple[float, float, float, float],
        save_path: str = "schedules.png"
    ) -> None:
        """
        Plots the LoRA and ControlNet multiplier schedules over the given number of steps.
        """
        l_s, l_e, l_m0, l_m1 = map(float, lora_schedule)
        c_s, c_e, c_m0, c_m1 = map(float, controlnet_schedule)

        l_vals = []
        c_vals = []
        steps = list(range(num_steps))

        for i in steps:
            progress = i / max(1, (num_steps - 1))
            
            l_mult = self._schedule_multiplier(
                progress, start_frac=l_s, end_frac=l_e, start_mult=l_m0, end_mult=l_m1
            )
            l_vals.append(l_mult)
            
            c_mult = self._schedule_multiplier(
                progress, start_frac=c_s, end_frac=c_e, start_mult=c_m0, end_mult=c_m1
            )
            c_vals.append(c_mult)

        plt.figure(figsize=(10, 5))
        plt.plot(steps, l_vals, label="LoRA Multiplier (Camera)", color='blue', linewidth=2)
        plt.plot(steps, c_vals, label="ControlNet Multiplier (Structure)", color='orange', linewidth=2)
        plt.title(f"Adapter Schedules over {num_steps} Steps")
        plt.xlabel("Step Index")
        plt.ylabel("Multiplier")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()


    # --------------------------
    # Main edit
    # --------------------------
    def run_simulation(
        self,
        *,
        source_image: Image.Image,
        camera: "CameraSettings",
        prompt: str,
        negative_prompt: str = (
            "ugly, duplicate, morbid, mutilated, out of frame, extra fingers, mutated hands, "
            "poorly drawn hands, poorly drawn face, mutation, deformed, blurry, bad anatomy, "
            "bad proportions, extra limbs, cloned face, disfigured, malformed limbs, missing arms, "
            "missing legs, extra legs, long neck"
        ),
        strength: float = 0.40,
        controlnet_scale: float = 0.8,
        lora_scale: float = 0.55,
        seed: Optional[int] = None,
        num_inference_steps: int = 40,
        guidance_scale: float = 7.5,
        control_image: Optional[Image.Image] = None,
        reference_image: Optional[Image.Image] = None,
        # Feature guidance knobs
        feat_weight: float = 1.5,
        feat_step_size: float = 0.06,
        feat_start_frac: float = 0.10,
        feat_end_frac: float = 0.55,
        feat_decay: Literal["linear", "cosine", "constant"] = "cosine",
        feat_every_k: int = 3,
        # ---- NEW: semantic feature mode (coarse, to avoid preserving bad details)
        semantic_mode: SemanticFeatureMode = "clip_global",
        # ---- NEW: Auto reference cleanup stage
        auto_ref_cleanup: bool = True,
        cleanup_prompt: Optional[str] = None,
        cleanup_negative_prompt: Optional[str] = None,
        cleanup_strength: float = 0.18,
        cleanup_steps: int = 20,
        cleanup_guidance_scale: float = 6.5,
        cleanup_lora_scale: float = 0.0,
        cleanup_controlnet_scale: float = 1.0,
        cleanup_use_camera_tokens: bool = False,
        # ---- NEW: LoRA scheduling (camera effect stronger late)
        lora_schedule: Tuple[float, float, float, float] = (0.0, 1.0, 0.30, 1.20),
        # ---- NEW: ControlNet scheduling (structure lock strong early, relaxed late)
        controlnet_schedule: Tuple[float, float, float, float] = (0.0, 1.0, 1.00, 0.35),
    ) -> Image.Image:
        """
        Same API style, with additional options:
          - auto_ref_cleanup: generate a "cleaner" reference for semantic guidance
          - semantic_mode: coarse feature guidance (default CLIP global)
          - lora_schedule/controlnet_schedule: temporal schedules
        """

        # Ensure we use the right backbone for semantic mode
        if semantic_mode == "clip_global":
            self.set_feature_backbone("clip")
        elif semantic_mode == "dino_cls":
            self.set_feature_backbone("dino")
        else:
            raise ValueError(f"Unknown semantic_mode: {semantic_mode}")

        source_image = self._rgb(source_image)

        # Prepare ControlNet condition image (depth) if needed
        if control_image is None and self.pipe.controlnet is not None:
            control_image = self._make_depth(source_image)
        if control_image is not None:
            control_image = self._resize_like(control_image, source_image)

        # Build a cleaned reference if requested and none provided
        if reference_image is None and auto_ref_cleanup:
            if cleanup_prompt is None:
                cleanup_prompt = (
                    "clean photo, natural lighting, correct exposure, sharp focus, low noise, "
                    "realistic colors, good dynamic range"
                )
            if cleanup_negative_prompt is None:
                cleanup_negative_prompt = negative_prompt

            reference_image = self._auto_cleanup_reference(
                source_image=source_image,
                prompt=cleanup_prompt,
                negative_prompt=cleanup_negative_prompt,
                control_image=control_image,
                cleanup_strength=cleanup_strength,
                cleanup_steps=cleanup_steps,
                cleanup_guidance_scale=cleanup_guidance_scale,
                cleanup_lora_scale=cleanup_lora_scale,
                cleanup_controlnet_scale=cleanup_controlnet_scale,
                seed=seed,
                use_camera_tokens=cleanup_use_camera_tokens,
                camera=camera,
                semantic_mode=semantic_mode,
            )
        elif reference_image is None:
            reference_image = source_image
        else:
            reference_image = self._resize_like(self._rgb(reference_image), source_image)

        reference_image.save("reference_image.png")

        # Precompute reference semantic feature (no grad)
        with torch.no_grad():
            ref_img01 = self._pil_to_img01(reference_image)  # [1,3,H,W] in [0,1]
            ref_feat = self._feat_semantic_from_img01(ref_img01, mode=semantic_mode)  # [1,D]

        # Prompt embeds + camera token injection for MAIN run (no grad)
        with torch.no_grad():
            prompt_embeds, negative_prompt_embeds = self.pipe.encode_prompt(
                prompt=prompt,
                device=self.device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
            )
            prompt_embeds, negative_prompt_embeds = embed_camera_settings(
                camera.focal_length,
                camera.f_number,
                camera.iso_speed_rating,
                camera.exposure_time,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                cam_embed=self.cam_embed,
                device=self.device,
            )

        # Timesteps / strength
        self.pipe.scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps = self.pipe.scheduler.timesteps

        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)
        t_start = max(num_inference_steps - init_timestep, 0)
        timesteps = timesteps[t_start:]

        # Plot schedules for this run
        self._plot_schedules(
            num_steps=len(timesteps),
            lora_schedule=lora_schedule,
            controlnet_schedule=controlnet_schedule,
            save_path="last_run_schedules.png"
        )


        # Prepare initial latents from source image
        generator = torch.Generator(device=self.pipe._execution_device)
        if seed is not None:
            generator.manual_seed(int(seed))

        init_image_tensor = self.pipe.image_processor.preprocess(source_image).to(
            self.device, dtype=self.torch_dtype
        )
        latents = self.pipe.prepare_latents(
            init_image_tensor,
            timestep=timesteps[0],
            batch_size=1,
            num_images_per_prompt=1,
            dtype=self.torch_dtype,
            device=self.device,
            generator=generator,
        )

        # Control tensor for the manual loop (UNet/ControlNet path)
        if control_image is not None:
            control_tensor = self.pipe.image_processor.preprocess(control_image).to(
                self.device, dtype=self.torch_dtype
            )
        else:
            control_tensor = None

        # Feature-guidance window
        n = len(timesteps)
        s0 = max(0, min(int(feat_start_frac * n), n - 1))
        s1 = max(s0 + 1, min(int(feat_end_frac * n), n))

        def weight_at(i: int) -> float:
            if i < s0 or i >= s1:
                return 0.0
            u = (i - s0) / max(1, (s1 - s0 - 1))
            if feat_decay == "linear":
                return float(feat_weight) * (1.0 - u)
            if feat_decay == "cosine":
                return float(feat_weight) * (0.5 * (1.0 + torch.cos(torch.tensor(u * 3.1415926535))).item())
            return float(feat_weight)

        # Parse schedules
        l_s, l_e, l_m0, l_m1 = map(float, lora_schedule)
        c_s, c_e, c_m0, c_m1 = map(float, controlnet_schedule)

        base_lora = float(lora_scale)
        base_ctrl = float(controlnet_scale)

        # Denoising loop
        for i, t in tqdm(enumerate(timesteps), total=len(timesteps), desc="Denoising"):
            progress = i / max(1, (len(timesteps) - 1))

            # Scheduled LoRA (camera stronger late)
            l_mult = self._schedule_multiplier(
                progress, start_frac=l_s, end_frac=l_e, start_mult=l_m0, end_mult=l_m1
            )
            self.set_scales(lora_scale=base_lora * l_mult)

            # Scheduled ControlNet (strong early, weaker late)
            c_mult = self._schedule_multiplier(
                progress, start_frac=c_s, end_frac=c_e, start_mult=c_m0, end_mult=c_m1
            )
            ctrl_scale_t = base_ctrl * c_mult

            # -----------------------------
            # 1) Normal denoising step: NO GRAD
            # -----------------------------
            with torch.no_grad():
                latent_in = torch.cat([latents] * 2)
                latent_in = self.pipe.scheduler.scale_model_input(latent_in, t)
                enc = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)

                down_samples = mid_sample = None
                if self.pipe.controlnet is not None and control_tensor is not None:
                    ctrl_in = torch.cat([control_tensor] * 2)
                    down_samples, mid_sample = self.pipe.controlnet(
                        latent_in,
                        t,
                        encoder_hidden_states=enc,
                        controlnet_cond=ctrl_in,
                        conditioning_scale=float(ctrl_scale_t),
                        return_dict=False,
                    )

                noise_pred = self.pipe.unet(
                    latent_in,
                    t,
                    encoder_hidden_states=enc,
                    down_block_additional_residuals=down_samples,
                    mid_block_additional_residual=mid_sample,
                    return_dict=False,
                )[0]

                n_uncond, n_text = noise_pred.chunk(2)
                noise_pred = n_uncond + float(guidance_scale) * (n_text - n_uncond)

                latents = self.pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

            # -----------------------------
            # 2) Coarse semantic PnP step: GRAD ONLY HERE
            # -----------------------------
            w = weight_at(i)
            if w > 0.0 and feat_step_size > 0.0 and (i % max(1, int(feat_every_k)) == 0):
                with torch.enable_grad():
                    latents_g = latents.detach().requires_grad_(True)

                    scaled = self.pipe.scheduler.scale_model_input(latents_g, t)
                    noise_c = self.pipe.unet(
                        scaled,
                        t,
                        encoder_hidden_states=prompt_embeds,
                        return_dict=False,
                    )[0]

                    step_out = self.pipe.scheduler.step(noise_c, t, latents_g, return_dict=True)
                    x0 = step_out.pred_original_sample if getattr(step_out, "pred_original_sample", None) is not None else latents_g

                    x0_vae = x0 / self.pipe.vae.config.scaling_factor
                    img = self.pipe.vae.decode(x0_vae, return_dict=False)[0]  # [-1,1]
                    img01 = (img / 2 + 0.5).clamp(0, 1)

                    feat = self._feat_semantic_from_img01(img01, mode=semantic_mode)  # [1,D]
                    loss = self._cosine_dist_loss(feat, ref_feat) * float(w)

                    grad = torch.autograd.grad(loss, latents_g, retain_graph=False, create_graph=False)[0]
                    grad = grad / (grad.norm(p=2) + 1e-8)

                    latents = (latents_g - float(feat_step_size) * grad).detach()
                    del latents_g, scaled, noise_c, step_out, x0, x0_vae, img, img01, feat, loss, grad

        # Decode final
        with torch.no_grad():
            z = latents / self.pipe.vae.config.scaling_factor
            img = self.pipe.vae.decode(z, return_dict=False)[0]
            img01 = (img / 2 + 0.5).clamp(0, 1)

        arr = img01[0].permute(1, 2, 0).cpu().numpy()
        return Image.fromarray((arr * 255).astype("uint8"))


# -------------------------
# Example usage
# -------------------------
if __name__ == "__main__":
    # engine = CameraAlignedEditingEngine(
    #     model_id="Manojb/stable-diffusion-2-1-base",
    #     controlnet_model_id="thibaud/controlnet-sd21-depth-diffusers",
    #     camera_setting_embedding_id="ishengfang/Camera-Settings-as-Tokens-SD2",
    #     device="cuda:0",
    # )

    engine = EnhancedPnPFeatureGuidedEditor(
        model_id="Manojb/stable-diffusion-2-1-base",
        controlnet_model_id="thibaud/controlnet-sd21-depth-diffusers",
        camera_setting_embedding_id="ishengfang/Camera-Settings-as-Tokens-SD2",
        device="cuda:1",
    )

    img = Image.open("/media/Pluto/alexpeng/DGIM/finalProject/src/demo_bad_images/DSC06264.jpg").convert("RGB")

    cam = CameraSettings(
        focal_length=50.0,
        f_number=8,
        iso_speed_rating=100.0,
        exposure_time=1/400,
    )

    sim = engine.run_simulation(
        source_image=img,
        camera=cam,
        prompt="photorealistic portrait, natural lighting",
        strength=0.35, # 0.3
        semantic_mode="clip_global",
        controlnet_scale=0.75,
        guidance_scale=8.5,
        lora_scale=0.55, #0.55
        seed=123,
    )
    sim.save("simulation_enhanced_clip_f8_105mm.png")
        

    # Faithful camera-aligned simulation
    # sim = engine.run_simulation(
    #     source_image=img,
    #     camera=cam,
    #     prompt="photorealistic portrait, natural lighting, shallow depth of field",
    #     strength=0.6, # 0.3
    #     ip_adapter_scale=0.7,
    #     controlnet_scale=0.8,
    #     lora_scale=0.55, #0.55
    #     seed=123,
    # )
    # sim.save("simulation.png")

    # Multiple variations
    # vars_ = engine.run_variation(
    #     source_image=img,
    #     camera=cam,
    #     prompt="photorealistic, natural lighting, shallow depth of field",
    #     num_variations=4,
    #     strength=0.55,
    #     ip_adapter_scale=0.55,
    #     controlnet_scale=0.75,
    #     lora_scale=0.55,
    # )
    # for i, v in enumerate(vars_):
    #     v.save(f"variation_{i}.png")

