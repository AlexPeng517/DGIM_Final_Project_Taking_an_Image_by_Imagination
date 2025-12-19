import os
import argparse
from dataclasses import asdict
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image.fid import FrechetInceptionDistance
from transformers import CLIPModel, CLIPProcessor


"""
evaluate_camera_aware_editing.py

Evaluates FOUR methods:
  1) baseline_depth_only            -> CameraAlignedEditingEngine (NO IP-Adapter)
  2) camera_token_clip_pnp          -> PnPFeatureGuidedEditor(feature_type="clip")
  3) camera_token_dino_pnp          -> PnPFeatureGuidedEditor(feature_type="dino")
  4) enhanced_cleanup_clip          -> EnhancedPnPFeatureGuidedEditor(feature_type="clip")

Camera sweep:
  - shallow DoF (tele + wide aperture)
  - deep DoF (wide + stopped down)
  - shutter/ISO variations

Metrics:
  - LPIPS (alex)
  - CLIP IA score (cosine similarity)
  - PSNR
  - FID (torchmetrics)

Reporting:
  - Per (method, camera_setting): mean and std across images
  - Per method: MACRO-average across camera settings
      * For each metric M:
          - macro_mean = mean_over_settings( M_setting_mean )
          - macro_std  = std_over_settings( M_setting_mean )
        (This reflects variability across camera settings, which is typically what you want
         when claiming robustness to camera changes.)
      * For FID (computed per-setting), macro_mean/std computed over per-setting FID.

Usage:
  python evaluate_camera_aware_editing.py \
    --input_dir /path/to/images \
    --output_dir /path/to/results \
    --device cuda:0 \
    --limit 50
"""


try:
    from engine.cameraEngine import (
        CameraAlignedEditingEngine,
        PnPFeatureGuidedEditor,
        EnhancedPnPFeatureGuidedEditor,
        CameraSettings,
    )
except ImportError:
    import sys
    current_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = current_dir if os.path.basename(current_dir) == "src" else os.path.dirname(current_dir)
    engine_dir = os.path.join(src_dir, "engine")
    if src_dir not in sys.path:
        sys.path.append(src_dir)
    if engine_dir not in sys.path:
        sys.path.append(engine_dir)

    from engine.cameraEngine import (
        CameraAlignedEditingEngine,
        PnPFeatureGuidedEditor,
        EnhancedPnPFeatureGuidedEditor,
        CameraSettings,
    )

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def list_images(input_dir: str, limit: Optional[int] = None) -> List[str]:
    files = [
        f for f in os.listdir(input_dir)
        if os.path.splitext(f)[1].lower() in VALID_EXTS
    ]
    files.sort()
    if limit is not None:
        files = files[: int(limit)]
    return files


def calculate_psnr(img1: Image.Image, img2: Image.Image) -> float:
    if img1.size != img2.size:
        img2 = img2.resize(img1.size, Image.BICUBIC)
    t1 = TF.to_tensor(img1)
    t2 = TF.to_tensor(img2)
    mse = torch.mean((t1 - t2) ** 2)
    if float(mse) == 0.0:
        return float("inf")
    return float(20.0 * torch.log10(1.0 / torch.sqrt(mse)))


def pil_to_uint8_299(pil: Image.Image, device: str) -> torch.Tensor:
    arr = np.array(pil)  # HWC uint8
    t = torch.from_numpy(arr).permute(2, 0, 1)  # CHW uint8
    t = TF.resize(t, [299, 299], antialias=True)
    return t.unsqueeze(0).to(device=device, dtype=torch.uint8)


def stable_seed(img_file: str, camera_name: str, base: int = 42) -> int:
    key = f"{img_file}|{camera_name}"
    return int(base + (abs(hash(key)) % 1_000_000))


def make_camera_sweep() -> List[Tuple[str, CameraSettings]]:
    return [
        ("shallow_dof_tele_fast",
         CameraSettings(focal_length=85.0, f_number=1.8, iso_speed_rating=100.0, exposure_time=1/200)),
        ("deep_dof_wide_stoppeddown",
         CameraSettings(focal_length=16.0, f_number=16.0, iso_speed_rating=100.0, exposure_time=1/60)),
        ("lowlight_highISO_slowShutter",
         CameraSettings(focal_length=35.0, f_number=2.8, iso_speed_rating=3200.0, exposure_time=1/20)),
        ("daylight_lowISO_fastShutter",
         CameraSettings(focal_length=35.0, f_number=2.8, iso_speed_rating=100.0, exposure_time=1/1000)),
    ]


def fmt_mean_std(mean: float, std: float, decimals: int = 4) -> str:
    if np.isinf(mean):
        return "inf"
    return f"{mean:.{decimals}f} ± {std:.{decimals}f}"


class Evaluator:
    def __init__(self, device: str = "cuda:0"):
        self.device = device
        print(f"Initializing metrics on {device}...")

        self.lpips = LearnedPerceptualImagePatchSimilarity(net_type="alex").to(device)

        self.clip_model = CLIPModel.from_pretrained(
            "openai/clip-vit-base-patch32",
            use_safetensors=True
        ).to(device).eval()
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        self.fid_metric = FrechetInceptionDistance(feature=2048).to(device)

    @torch.no_grad()
    def compute_pair_metrics(self, original_pil: Image.Image, generated_pil: Image.Image) -> Dict[str, float]:
        if original_pil.size != generated_pil.size:
            generated_pil = generated_pil.resize(original_pil.size, Image.BICUBIC)

        t_orig = TF.to_tensor(original_pil).unsqueeze(0).to(self.device)
        t_gen = TF.to_tensor(generated_pil).unsqueeze(0).to(self.device)

        lpips_val = float(self.lpips((t_gen * 2 - 1).clamp(-1, 1), (t_orig * 2 - 1).clamp(-1, 1)).item())

        inputs_orig = self.clip_processor(images=original_pil, return_tensors="pt").to(self.device)
        inputs_gen = self.clip_processor(images=generated_pil, return_tensors="pt").to(self.device)
        feat_orig = self.clip_model.get_image_features(**inputs_orig)
        feat_gen = self.clip_model.get_image_features(**inputs_gen)
        ia_score = float(F.cosine_similarity(feat_orig, feat_gen).item())

        psnr_val = float(calculate_psnr(original_pil, generated_pil))

        return {"lpips": lpips_val, "ia_score": ia_score, "psnr": psnr_val}

    @torch.no_grad()
    def fid_reset(self):
        self.fid_metric.reset()

    @torch.no_grad()
    def fid_update_pair(self, real_pil: Image.Image, gen_pil: Image.Image):
        self.fid_metric.update(pil_to_uint8_299(real_pil, self.device), real=True)
        self.fid_metric.update(pil_to_uint8_299(gen_pil, self.device), real=False)

    @torch.no_grad()
    def fid_compute(self) -> float:
        return float(self.fid_metric.compute().item())


def build_engine(method_name: str, device: str):
    if method_name == "baseline_depth_only":
        return CameraAlignedEditingEngine(device=device)

    if method_name == "camera_token_clip_pnp":
        return PnPFeatureGuidedEditor(feature_type="clip", device=device, torch_dtype=torch.float16)

    if method_name == "camera_token_dino_pnp":
        return PnPFeatureGuidedEditor(feature_type="dino", device=device, torch_dtype=torch.float16)

    if method_name == "enhanced_cleanup_clip":
        return EnhancedPnPFeatureGuidedEditor(feature_type="clip", device=device, torch_dtype=torch.float16)

    raise ValueError(f"Unknown method_name: {method_name}")


def macro_aggregate_per_method(per_setting_rows: List[Dict]) -> List[Dict]:
    """
    Macro-average across camera settings:
      - For lpips/ia_score/psnr: macro over the per-setting MEANS
      - For fid: macro over the per-setting fid values
    Also provide macro std across settings (std of per-setting means).
    """
    df = pd.DataFrame(per_setting_rows)

    out_rows = []
    for method, g in df.groupby("method", sort=False):
        row = {"method": method, "camera_setting": "MACRO_AVG_OVER_SETTINGS", "num_settings": int(len(g))}

        # Metrics: macro mean/std computed over per-setting means
        for m in ["lpips", "ia_score", "psnr"]:
            vals = g[f"{m}_mean"].to_numpy(dtype=np.float64)
            row[f"{m}_macro_mean"] = float(np.mean(vals))
            row[f"{m}_macro_std"] = float(np.std(vals, ddof=0))

        # FID: macro mean/std over per-setting FID
        fid_vals = g["fid"].to_numpy(dtype=np.float64)
        row["fid_macro_mean"] = float(np.mean(fid_vals))
        row["fid_macro_std"] = float(np.std(fid_vals, ddof=0))

        out_rows.append(row)

    return out_rows


def run_evaluation(args):
    os.makedirs(args.output_dir, exist_ok=True)
    image_files = list_images(args.input_dir, args.limit)
    print(f"Found {len(image_files)} images in {args.input_dir}")

    camera_sweep = make_camera_sweep()

    prompt = "a perfect photo with perfect shot"
    negative_prompt = (
        "ugly, duplicate, morbid, mutilated, out of frame, extra fingers, mutated hands, "
        "poorly drawn hands, poorly drawn face, mutation, deformed, blurry, bad anatomy, "
        "bad proportions, extra limbs, cloned face, disfigured, malformed limbs, missing arms, "
        "missing legs, extra legs, long neck"
    )

    methods = [
        "baseline_depth_only",
        "camera_token_clip_pnp",
        "camera_token_dino_pnp",
        "enhanced_cleanup_clip",
    ]

    evaluator = Evaluator(device=args.device)

    per_setting_rows: List[Dict] = []
    macro_rows: List[Dict] = []

    for method_name in methods:
        print(f"\n=== Method: {method_name} ===")
        engine = build_engine(method_name, device=args.device)

        method_out_dir = os.path.join(args.output_dir, method_name)
        os.makedirs(method_out_dir, exist_ok=True)

        for cam_name, cam in camera_sweep:
            print(f"\n--- Camera setting: {cam_name} | {asdict(cam)} ---")
            cam_out_dir = os.path.join(method_out_dir, cam_name)
            os.makedirs(cam_out_dir, exist_ok=True)

            evaluator.fid_reset()
            metrics_acc = {"lpips": [], "ia_score": [], "psnr": []}

            for img_file in tqdm(image_files, desc=f"{method_name} | {cam_name}"):
                src_path = os.path.join(args.input_dir, img_file)
                original = load_image(src_path)
                seed = stable_seed(img_file, cam_name, base=int(args.seed))

                # Generate
                if method_name == "baseline_depth_only":
                    gen = engine.run_simulation(
                        source_image=original,
                        camera=cam,
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        strength=0.40,
                        controlnet_scale=0.75,
                        lora_scale=0.55,
                        seed=seed,
                    )

                elif method_name in ("camera_token_clip_pnp", "camera_token_dino_pnp"):
                    gen = engine.run_simulation(
                        source_image=original,
                        camera=cam,
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        strength=0.40,
                        controlnet_scale=0.75,
                        lora_scale=0.55,
                        seed=seed,
                        num_inference_steps=40,
                        guidance_scale=7.5,
                        feat_weight=0.7,
                        feat_step_size=0.06,
                        feat_start_frac=0.10,
                        feat_end_frac=0.55,
                        feat_decay="cosine",
                        feat_every_k=3,
                    )

                elif method_name == "enhanced_cleanup_clip":
                    gen = engine.run_simulation(
                        source_image=original,
                        camera=cam,
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        strength=0.40,
                        controlnet_scale=0.75,
                        lora_scale=0.55,
                        seed=seed,
                        num_inference_steps=40,
                        guidance_scale=7.5,

                        semantic_mode="clip_global",
                        auto_ref_cleanup=True,
                        cleanup_prompt="clean photo, natural lighting, correct exposure, sharp focus, low noise, realistic colors",
                        cleanup_strength=0.18,
                        cleanup_steps=20,
                        cleanup_guidance_scale=6.5,
                        cleanup_lora_scale=0.0,
                        cleanup_controlnet_scale=1.0,
                        cleanup_use_camera_tokens=False,

                        feat_weight=1.2,
                        feat_step_size=0.06,
                        feat_start_frac=0.10,
                        feat_end_frac=0.55,
                        feat_decay="cosine",
                        feat_every_k=3,

                        lora_schedule=(0.0, 1.0, 0.30, 1.20),
                        controlnet_schedule=(0.0, 1.0, 1.00, 0.35),
                    )

                else:
                    raise RuntimeError("unreachable")

                # Save
                save_path = os.path.join(cam_out_dir, img_file)
                gen.save(save_path)

                # Metrics
                pair = evaluator.compute_pair_metrics(original, gen)
                for k in metrics_acc:
                    metrics_acc[k].append(pair[k])

                evaluator.fid_update_pair(original, gen)

                del original, gen

            fid = evaluator.fid_compute()

            # Per (method, setting) summary
            row = {
                "method": method_name,
                "camera_setting": cam_name,
                "num_images": len(image_files),
                "fid": float(fid),
            }
            for k, vals in metrics_acc.items():
                row[f"{k}_mean"] = float(np.mean(vals))
                row[f"{k}_std"] = float(np.std(vals, ddof=0))

            per_setting_rows.append(row)

            print("Result (per setting):")
            print(f"  LPIPS:   {fmt_mean_std(row['lpips_mean'], row['lpips_std'])}")
            print(f"  IA:      {fmt_mean_std(row['ia_score_mean'], row['ia_score_std'])}")
            print(f"  PSNR:    {fmt_mean_std(row['psnr_mean'], row['psnr_std'])}")
            print(f"  FID:     {row['fid']:.4f}")

        # macro-average for this method (over its settings)
        macro = macro_aggregate_per_method([r for r in per_setting_rows if r["method"] == method_name])[0]
        macro_rows.append(macro)

        print("\nMacro-average over camera settings (per-setting means):")
        print(f"  LPIPS:   {fmt_mean_std(macro['lpips_macro_mean'], macro['lpips_macro_std'])}")
        print(f"  IA:      {fmt_mean_std(macro['ia_score_macro_mean'], macro['ia_score_macro_std'])}")
        print(f"  PSNR:    {fmt_mean_std(macro['psnr_macro_mean'], macro['psnr_macro_std'])}")
        print(f"  FID:     {fmt_mean_std(macro['fid_macro_mean'], macro['fid_macro_std'])}")

        del engine
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Save reports
    df_setting = pd.DataFrame(per_setting_rows)
    df_macro = pd.DataFrame(macro_rows)

    csv_setting = os.path.join(args.output_dir, "evaluation_report_per_setting.csv")
    csv_macro = os.path.join(args.output_dir, "evaluation_report_macro_over_settings.csv")

    df_setting.to_csv(csv_setting, index=False)
    df_macro.to_csv(csv_macro, index=False)

    # Consolidated display table (professional style)
    display_df = df_macro[[
        "method", "num_settings",
        "lpips_macro_mean", "lpips_macro_std",
        "ia_score_macro_mean", "ia_score_macro_std",
        "psnr_macro_mean", "psnr_macro_std",
        "fid_macro_mean", "fid_macro_std",
    ]].copy()

    print(f"\nSaved per-setting report: {csv_setting}")
    print(f"Saved macro report:       {csv_macro}")

    print("\n=== Macro report (mean ± std across camera settings) ===")
    for _, r in display_df.iterrows():
        print(f"\nMethod: {r['method']}  (num_settings={int(r['num_settings'])})")
        print(f"  LPIPS: {fmt_mean_std(r['lpips_macro_mean'], r['lpips_macro_std'])}")
        print(f"  IA:    {fmt_mean_std(r['ia_score_macro_mean'], r['ia_score_macro_std'])}")
        print(f"  PSNR:  {fmt_mean_std(r['psnr_macro_mean'], r['psnr_macro_std'])}")
        print(f"  FID:   {fmt_mean_std(r['fid_macro_mean'], r['fid_macro_std'])}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate camera-aware editing engines with camera sweep + macro average")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing source images")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save outputs + reports")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of images")
    parser.add_argument("--seed", type=int, default=42, help="Base seed (per-image derived deterministically)")
    args = parser.parse_args()
    run_evaluation(args)

