# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
#
# import argparse
# import os
# import sys
# from pathlib import Path
# from typing import List, Tuple, Union, Dict
#
# import numpy as np
# from PIL import Image, ImageOps
#
# import torch
# from torchvision import transforms
# from tqdm import tqdm
#
# from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
# from transformers import CLIPTextModel, CLIPTokenizer
#
# import yaml
# import importlib
#
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# DEFAULT_SCALING = 0.18215
# IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
#
# # --- 新增：从 yaml 配置动态实例化模型 ---
# def instantiate_from_target(target: str, params: dict):
#     """
#     target: full path like ldm.modules.diffusionmodules.openaimodel_inject.UNetModel_inject
#     params: dict of kwargs to pass to the class
#     """
#     module_name, class_name = target.rsplit(".", 1)
#     mod = importlib.import_module(module_name)
#     cls = getattr(mod, class_name)
#     # shallow copy of params to avoid mutating caller
#     p = dict(params or {})
#     return cls(**p)
#
# def load_models_from_yaml(yaml_path: str, ckpt_map: dict = None):
#     ckpt_map = ckpt_map or {}
#     with open(yaml_path, "r", encoding="utf-8") as f:
#         cfg = yaml.safe_load(f)
#     model_cfg = cfg.get("model", {}).get("params", {})
#     result = {}
#     # UNET
#     if "unet_config" in model_cfg:
#         try:
#             unet_conf = model_cfg["unet_config"]
#             target = unet_conf["target"]
#             params = unet_conf.get("params", {}) or {}
#             unet = instantiate_from_target(target, params)
#             unet = unet.to(DEVICE).eval()
#             if "unet" in ckpt_map and ckpt_map["unet"]:
#                 state = torch.load(ckpt_map["unet"], map_location=DEVICE)
#                 sd = state.get("state_dict", state)
#                 unet.load_state_dict(sd, strict=False)
#             result["unet"] = unet
#         except Exception as e:
#             print(f"[Warn] instantiate unet from cfg failed: {e}", file=sys.stderr)
#
#     # VAE / first_stage
#     if "first_stage_config" in model_cfg:
#         try:
#             vae_conf = model_cfg["first_stage_config"]
#             target = vae_conf["target"]
#             params = vae_conf.get("params", {}) or {}
#             vae = instantiate_from_target(target, params)
#             vae = vae.to(DEVICE).eval()
#             if "vae" in ckpt_map and ckpt_map["vae"]:
#                 state = torch.load(ckpt_map["vae"], map_location=DEVICE)
#                 sd = state.get("state_dict", state)
#                 vae.load_state_dict(sd, strict=False)
#             result["vae"] = vae
#         except Exception as e:
#             print(f"[Warn] instantiate vae from cfg failed: {e}", file=sys.stderr)
#
#     # expose raw cfg for possible cond_stage info (e.g., clip version)
#     result["_cfg"] = cfg
#     return result
#
#
# # ---------- image preprocess: keep aspect, shortest side -> target, round to /8 ----------
# # def resize_shortest_keep_aspect(pil_image: Image.Image, target_short: int = 512) -> Image.Image:
# #     w, h = pil_image.size
# #     # s = target_short / float(min(h, w))
# #     s = 1
# #     new_w = int(round((w * s) / 8.0) * 8)
# #     new_h = int(round((h * s) / 8.0) * 8)
# #     new_w = max(new_w, 8)
# #     new_h = max(new_h, 8)
# #     return pil_image.resize((new_w, new_h), resample=Image.BICUBIC)
#
#
# def resize_shortest_keep_aspect(pil_image: Image.Image, mult: int = 8) -> Image.Image:
#     # 矫正EXIF方向，避免横竖颠倒
#     img = ImageOps.exif_transpose(pil_image).convert("RGB")
#     w, h = img.size
#
#     # 已经是 /mult 就直接返回，完全不插值
#     if w % mult == 0 and h % mult == 0:
#         return img
#
#     # 只做右/下方向的零填充到最近的 /mult，不缩放
#     new_w = ((w + mult - 1) // mult) * mult
#     new_h = ((h + mult - 1) // mult) * mult
#     pad_r, pad_b = new_w - w, new_h - h
#     # 边界值填充：边缘像素复制，避免黑边（可改为常数/反射等）
#     return ImageOps.expand(img, border=(0, 0, pad_r, pad_b), fill=img.getpixel((w-1, h-1)))
#
#
#
# def load_image_tensor(path: Path, target_short: int = 256):
#     # type: (Path, int) -> Tuple[torch.Tensor, Tuple[int, int], Tuple[int, int]]
#     img = Image.open(path).convert("RGB")
#     H0, W0 = img.height, img.width
#     img_rs = resize_shortest_keep_aspect(img, target_short)
#     Hc, Wc = img_rs.height, img_rs.width
#
#     tfm = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize([0.5] * 3, [0.5] * 3)  # -> [-1,1]
#     ])
#     x = tfm(img_rs).unsqueeze(0)  # [1,3,Hc,Wc]
#     return x, (H0, W0), (Hc, Wc)
#
#
# @torch.inference_mode()
# def extract_multiscale_latent_feats(
#     vae, unet, scheduler, img_tensor, t_list, cond_emb, dtype=torch.float16
# ):
#     img_tensor = img_tensor.to(DEVICE, dtype=torch.float32)
#     z = vae.encode(img_tensor).latent_dist.mean
#     scaling = getattr(vae.config, "scaling_factor", 0.18215)
#     z = (z * scaling).to(dtype=dtype)
#
#     # ---- NEW: 统一把 hook 输出转成 Tensor ----
#     def _as_tensor(o):
#         # 常见三种：Tensor / dataclass(带 hidden_states 或 sample) / tuple(list)
#         if isinstance(o, torch.Tensor):
#             return o
#         if hasattr(o, "hidden_states"):
#             o = o.hidden_states
#         elif hasattr(o, "sample"):
#             o = o.sample
#         elif isinstance(o, (tuple, list)) and len(o) > 0:
#             o = o[0]
#         if not isinstance(o, torch.Tensor):
#             raise TypeError(f"Hook output is not Tensor: {type(o)}")
#         return o
#
#     cache = {"p8_raw": None, "db0": None, "db1": None}
#     def hook_p8(_m, _i, o):  cache["p8_raw"] = o
#     def hook_db0(_m, _i, o): cache["db0"] = o
#     def hook_db1(_m, _i, o): cache["db1"] = o
#
#     h1 = unet.conv_in.register_forward_hook(hook_p8)
#     h2 = unet.down_blocks[0].register_forward_hook(hook_db0)
#     h3 = unet.down_blocks[1].register_forward_hook(hook_db1)
#
#     acc = {"p8": [], "p16": [], "p32": []}
#     T = scheduler.config.num_train_timesteps
#
#     for t in t_list:
#         t_int = int(t)
#         if not (0 <= t_int < T):
#             raise ValueError(f"t={t_int} out of range [0,{T-1}]")
#
#         cache["p8_raw"] = cache["db0"] = cache["db1"] = None
#
#         noise = torch.randn_like(z)
#         t_tensor = torch.tensor([t_int], device=DEVICE, dtype=torch.long)
#         zt = scheduler.add_noise(z, noise, t_tensor)
#         _ = unet(zt, t_tensor, encoder_hidden_states=cond_emb)
#
#         # ---- NEW: 先标准化为 Tensor，再 squeeze/cast ----
#         f8  = _as_tensor(cache["p8_raw"]).squeeze(0).to(dtype)
#         fA  = _as_tensor(cache["db0"]).squeeze(0).to(dtype)
#         fB  = _as_tensor(cache["db1"]).squeeze(0).to(dtype)
#
#         # 用分辨率判定谁是 p16/p32（相对 p8 约 /2、/4）
#         Hp8, Wp8 = f8.shape[-2:]
#         def score(fr, r):  # r=2 for p16, r=4 for p32
#             h, w = fr.shape[-2], fr.shape[-1]
#             return abs(Hp8/max(h,1)-r) + abs(Wp8/max(w,1)-r)
#         if score(fA, 2) + score(fB, 4) <= score(fA, 4) + score(fB, 2):
#             f16, f32 = fA, fB
#         else:
#             f16, f32 = fB, fA
#
#         acc["p8"].append(f8)
#         acc["p16"].append(f16)
#         acc["p32"].append(f32)
#
#     h1.remove(); h2.remove(); h3.remove()
#     feats_avg = {k: torch.stack(v, 0).mean(0).contiguous() for k, v in acc.items()}
#     return feats_avg
#
#
# def ensure_sd_root(sd_arg):
#     # type: (str) -> Union[str, Path]
#     """
#     Accept HF repo id or a local dir. If local dir, require subfolders exist.
#     We'll load all components via from_pretrained(sd_root, subfolder="...").
#     """
#     p = Path(sd_arg)
#     if p.exists() and p.is_dir():
#         needed = ["unet", "scheduler", "tokenizer", "text_encoder"]
#         missing = [d for d in needed if not (p / d).is_dir()]
#         if missing:
#             raise FileNotFoundError(f"Local SD dir missing subfolders {missing} in {p}")
#         return p
#     return sd_arg  # treat as HF repo id
#
#
# def list_images(img_dir: Path) -> List[Path]:
#     files = []
#     for name in sorted(os.listdir(img_dir)):
#         p = img_dir / name
#         if p.is_file() and p.suffix.lower() in IMG_EXTS:
#             files.append(p)
#     return files
#
#
# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--img-dir", type=str, required=True, help="Folder of images (COCO train2017/val2017 etc.)")
#     ap.add_argument("--out-dir", type=str, required=True, help="Where to save .pt feature files")
#     ap.add_argument("--sd", type=str, required=True,
#                     help="HF repo id OR local SD root containing: unet/, scheduler/, tokenizer/, text_encoder/, vae/")
#     ap.add_argument("--cfg", type=str, default=None,
#                     help="Optional: path to YAML config (e.g. unet-finetune-master/configs/.../attn2o-ffni.yaml). If provided, will instantiate unet/vae from it when possible.")
#     ap.add_argument("--vae", type=str, default=None,
#                     help="Optional: VAE repo id OR local path to vae/ subfolder. If None, will try sd_root/subfolder=vae.")
#     ap.add_argument("--image-size", type=int, default=512,
#                     help="Target shortest side (keep aspect). We round H,W to /8.")
#     ap.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "fp32"])
#     ap.add_argument("--t-list", type=str, default="400,600",
#                     help="Comma-separated timesteps, e.g., '200,250,300'")
#     ap.add_argument("--overwrite", action="store_true", help="Overwrite existing .pt")
#     args = ap.parse_args()
#
#     img_dir = Path(args.img_dir)
#     out_dir = Path(args.out_dir)
#     out_dir.mkdir(parents=True, exist_ok=True)
#
#     sd_root = ensure_sd_root(args.sd)
#
#     # Try to load models from YAML cfg if provided
#     models_from_cfg = {}
#     if args.cfg is not None:
#         print(f"[Load] Try instantiate models from cfg: {args.cfg}")
#         try:
#             models_from_cfg = load_models_from_yaml(args.cfg)
#         except Exception as e:
#             print(f"[Warn] load_models_from_yaml failed: {e}", file=sys.stderr)
#             models_from_cfg = {}
#
#     # VAE: 优先使用 cfg 中的实例化结果，否则回退到 from_pretrained
#     if "vae" in models_from_cfg:
#         vae = models_from_cfg["vae"]
#         print("[Info] Using VAE instantiated from YAML config")
#     else:
#         print("[Load] VAE:", args.vae if args.vae is not None else f"{sd_root}/vae")
#         try:
#             if args.vae is not None:
#                 vae = AutoencoderKL.from_pretrained(args.vae).to(DEVICE).eval()
#             else:
#                 raise ValueError("force fallback")
#         except Exception:
#             vae = AutoencoderKL.from_pretrained(sd_root, subfolder="vae").to(DEVICE).eval()
#
#     # UNET: 优先使用 cfg 中的实例化结果，否则回退到 from_pretrained
#     if "unet" in models_from_cfg:
#         unet = models_from_cfg["unet"]
#         print("[Info] Using UNet instantiated from YAML config")
#     else:
#         print("[Load] UNet:", sd_root, "(subfolder=unet)")
#         unet = UNet2DConditionModel.from_pretrained(sd_root, subfolder="unet").to(DEVICE).eval()
#
#     # Scheduler: 仍按 sd_root 加载（config 通常不包含 scheduler）
#     print("[Load] Scheduler:", sd_root, "(subfolder=scheduler)")
#     scheduler = DDPMScheduler.from_pretrained(sd_root, subfolder="scheduler")
#
#     # Tokenizer & TextEncoder:
#     # 如果 cfg 中包含 cond_stage_config 且其中有 version 字段，尝试用该 version 加载 HF CLIP tokenizer/text encoder，
#     # 否则回退到 sd_root 下的 tokenizer/text_encoder.
#     tokenizer = None
#     text_encoder = None
#     used_cfg = models_from_cfg.get("_cfg", {})
#     cond_cfg = None
#     try:
#         cond_cfg = used_cfg.get("model", {}).get("params", {}).get("cond_stage_config", {})
#         clip_version = cond_cfg.get("params", {}).get("version") if cond_cfg else None
#     except Exception:
#         clip_version = None
#
#     if clip_version:
#         try:
#             print(f"[Load] Tokenizer & TextEncoder from cond_stage version: {clip_version}")
#             tokenizer = CLIPTokenizer.from_pretrained(clip_version)
#             text_encoder = CLIPTextModel.from_pretrained(clip_version).to(DEVICE).eval()
#         except Exception as e:
#             print(f"[Warn] load CLIP from cond_stage.version failed: {e}", file=sys.stderr)
#             tokenizer = None
#             text_encoder = None
#
#     if tokenizer is None or text_encoder is None:
#         print("[Load] Tokenizer & TextEncoder:", sd_root, "(subfolders=tokenizer,text_encoder)")
#         tokenizer = CLIPTokenizer.from_pretrained(sd_root, subfolder="tokenizer")
#         text_encoder = CLIPTextModel.from_pretrained(sd_root, subfolder="text_encoder").to(DEVICE).eval()
#
#     # build unconditional (empty) text embedding once
#     tokens = tokenizer(
#         [""], padding="max_length",
#         max_length=tokenizer.model_max_length,
#         return_tensors="pt",
#     )
#     with torch.no_grad():
#         uncond_emb = text_encoder(tokens.input_ids.to(DEVICE))[0]  # [1,77,hidden]
#
#     # unify dtype
#     dtype = torch.float16 if args.dtype == "fp16" else torch.float32
#     unet = unet.to(device=DEVICE, dtype=dtype)
#     uncond_emb = uncond_emb.to(dtype)
#
#     # parse t_list
#     t_list = [int(x) for x in args.t_list.split(",") if x.strip() != ""]
#     if len(t_list) == 0:
#         raise ValueError("Empty --t-list")
#
#     images = list_images(img_dir)
#     if not images:
#         print(f"[Warn] No images found in {img_dir}")
#         return
#
#     print(f"[Info] {len(images)} images found. Saving to: {out_dir}")
#     for img_path in tqdm(images, ncols=100):
#         out_path = out_dir / (img_path.stem + ".pt")
#         if out_path.exists() and not args.overwrite:
#             continue
#
#         try:
#             x, (H0, W0), (Hc, Wc) = load_image_tensor(img_path, args.image_size)
#             feats = extract_multiscale_latent_feats(
#                 vae, unet, scheduler, x, t_list, uncond_emb, dtype=dtype
#             )  # dict with 'p8','p16','p32'
#
#             p8, p16, p32 = feats["p8"], feats["p16"], feats["p32"]
#             C8, Hp8, Wp8 = p8.shape
#             C16, Hp16, Wp16 = p16.shape
#             C32, Hp32, Wp32 = p32.shape
#
#             payload = {
#                 "features": [p8, p16, p32],
#                 "order": ["p8", "p16", "p32"],
#                 "shapes": {
#                     "p8":  (int(C8),  int(Hp8),  int(Wp8)),
#                     "p16": (int(C16), int(Hp16), int(Wp16)),
#                     "p32": (int(C32), int(Hp32), int(Wp32)),
#                 },
#                 "orig_size": (int(H0), int(W0)),
#                 "canonical_size": (int(Hc), int(Wc)),
#                 "grid_h8": (int(Hp8), int(Wp8)),
#                 "t_used": [int(t) for t in t_list],
#                 "sd_repo": str(args.sd),
#                 "vae_repo": str(args.vae) if args.vae is not None else f"{sd_root}/vae",
#                 "dtype": "fp16" if dtype == torch.float16 else "fp32",
#             }
#             torch.save(payload, out_path)
#         except Exception as e:
#             print(f"[Error] {img_path}: {e}", file=sys.stderr)
#
#     print("[Done]")
#
#
# if __name__ == "__main__":
#     main()
# python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import sys
from pathlib import Path
from typing import List, Tuple, Union, Dict

import numpy as np
from PIL import Image, ImageOps

import torch
from torchvision import transforms
from tqdm import tqdm

from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer

import yaml
import importlib

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_SCALING = 0.18215
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def instantiate_from_target(target: str, params: dict):
    """
    Instantiate class from full target string like 'ldm.models...Class'
    """
    module_name, cls_name = target.rsplit(".", 1)
    module = importlib.import_module(module_name)
    cls = getattr(module, cls_name)
    if params is None:
        params = {}
    return cls(**params)


def load_models_from_yaml(yaml_path: str, ckpt_map: dict = None):
    """
    Read YAML config (same style as your attn2o-ffni.yaml), instantiate unet/vae if present.
    Returns dict possibly containing 'unet', 'vae', and raw '_cfg' for cond_stage info.
    """
    res = {}
    with open(yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model_cfg = cfg.get("model", {}).get("params", {})
    # UNET
    if "unet_config" in model_cfg:
        unet_cfg = model_cfg["unet_config"]
        target = unet_cfg.get("target")
        params = unet_cfg.get("params", {})
        try:
            res["unet"] = instantiate_from_target(target, params)
        except Exception as e:
            print(f"[Warn] instantiate unet from yaml failed: {e}")
    # VAE / first_stage
    if "first_stage_config" in model_cfg:
        fs_cfg = model_cfg["first_stage_config"]
        target = fs_cfg.get("target")
        params = fs_cfg.get("params", {})
        try:
            res["vae"] = instantiate_from_target(target, params)
        except Exception as e:
            print(f"[Warn] instantiate vae from yaml failed: {e}")

    res["_cfg"] = cfg
    return res


def resize_shortest_keep_aspect(pil_image: Image.Image, mult: int = 8) -> Image.Image:
    img = ImageOps.exif_transpose(pil_image).convert("RGB")
    w, h = img.size
    if w % mult == 0 and h % mult == 0:
        return img
    new_w = ((w + mult - 1) // mult) * mult
    new_h = ((h + mult - 1) // mult) * mult
    pad_r, pad_b = new_w - w, new_h - h
    return ImageOps.expand(img, border=(0, 0, pad_r, pad_b), fill=img.getpixel((w-1, h-1)))


def load_image_tensor(path: Path, target_short: int = 256):
    img = Image.open(path).convert("RGB")
    H0, W0 = img.height, img.width
    img_rs = resize_shortest_keep_aspect(img, target_short)
    Hc, Wc = img_rs.height, img_rs.width

    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    x = tfm(img_rs).unsqueeze(0)  # [1,3,Hc,Wc]
    return x, (H0, W0), (Hc, Wc)


@torch.inference_mode()
def extract_multiscale_latent_feats(
    vae, unet, scheduler, img_tensor, t_list, cond_emb, dtype=torch.float16
):
    img_tensor = img_tensor.to(DEVICE, dtype=torch.float32)
    z = vae.encode(img_tensor).latent_dist.mean
    scaling = getattr(vae.config, "scaling_factor", 0.18215)
    z = (z * scaling).to(dtype=dtype)

    # ---- NEW: 统一把 hook 输出转成 Tensor ----
    def _as_tensor(o):
        # 常见三种：Tensor / dataclass(带 hidden_states 或 sample) / tuple(list)
        if isinstance(o, torch.Tensor):
            return o
        if hasattr(o, "hidden_states"):
            o = o.hidden_states
        elif hasattr(o, "sample"):
            o = o.sample
        elif isinstance(o, (tuple, list)) and len(o) > 0:
            o = o[0]
        if not isinstance(o, torch.Tensor):
            raise TypeError(f"Hook output is not Tensor: {type(o)}")
        return o

    cache = {"p8_raw": None, "db0": None, "db1": None}
    def hook_p8(_m, _i, o):  cache["p8_raw"] = o
    def hook_db0(_m, _i, o): cache["db0"] = o
    def hook_db1(_m, _i, o): cache["db1"] = o

    roots = [unet, getattr(unet, "model", None), getattr(unet, "diffusion_model", None)]
    conv_module = None
    down_blocks_mod = None

    for r in roots:
        if r is None:
            continue
        if hasattr(r, "conv_in") and conv_module is None:
            conv_module = getattr(r, "conv_in")
        if hasattr(r, "down_blocks") and getattr(r, "down_blocks") is not None and len(getattr(r, "down_blocks")) >= 2:
            down_blocks_mod = getattr(r, "down_blocks")
        if conv_module is not None and down_blocks_mod is not None:
            break

    # fallback: try to find any list/tuple attribute of a root that looks like down blocks
    if down_blocks_mod is None:
        for r in roots:
            if r is None:
                continue
            for name in dir(r):
                try:
                    attr = getattr(r, name)
                except Exception:
                    continue
                if isinstance(attr, (list, tuple)) and len(attr) >= 2:
                    if all(hasattr(x, "register_forward_hook") for x in attr[:2]):
                        down_blocks_mod = attr
                        break
            if down_blocks_mod is not None:
                break

    # final check and register
    if conv_module is not None and down_blocks_mod is not None:
        h1 = conv_module.register_forward_hook(hook_p8)
        h2 = down_blocks_mod[0].register_forward_hook(hook_db0)
        h3 = down_blocks_mod[1].register_forward_hook(hook_db1)
    else:
        raise RuntimeError(
            "unet does not expose expected attributes for hook extraction; check YAML UNet class (expected conv_in + down_blocks or similar).")

    acc = {"p8": [], "p16": [], "p32": []}
    T = scheduler.config.num_train_timesteps

    for t in t_list:
        t_int = int(t)
        if not (0 <= t_int < T):
            raise ValueError(f"t={t_int} out of range [0,{T-1}]")

        cache["p8_raw"] = cache["db0"] = cache["db1"] = None

        noise = torch.randn_like(z)
        t_tensor = torch.tensor([t_int], device=DEVICE, dtype=torch.long)
        zt = scheduler.add_noise(z, noise, t_tensor)
        _ = unet(zt, t_tensor, encoder_hidden_states=cond_emb)

        # ---- NEW: 先标准化为 Tensor，再 squeeze/cast ----
        f8  = _as_tensor(cache["p8_raw"]).squeeze(0).to(dtype)
        fA  = _as_tensor(cache["db0"]).squeeze(0).to(dtype)
        fB  = _as_tensor(cache["db1"]).squeeze(0).to(dtype)

        # 用分辨率判定谁是 p16/p32（相对 p8 约 /2、/4）
        Hp8, Wp8 = f8.shape[-2:]
        def score(fr, r):  # r=2 for p16, r=4 for p32
            h, w = fr.shape[-2], fr.shape[-1]
            return abs(Hp8/max(h,1)-r) + abs(Wp8/max(w,1)-r)
        if score(fA, 2) + score(fB, 4) <= score(fA, 4) + score(fB, 2):
            f16, f32 = fA, fB
        else:
            f16, f32 = fB, fA

        acc["p8"].append(f8)
        acc["p16"].append(f16)
        acc["p32"].append(f32)

    h1.remove(); h2.remove(); h3.remove()
    feats_avg = {k: torch.stack(v, 0).mean(0).contiguous() for k, v in acc.items()}
    return feats_avg


def ensure_sd_root(sd_arg):
    """
    Accept HF repo id or a local dir. If local dir, require subfolders exist.
    Return path string (repo id or local dir).
    """
    p = Path(sd_arg)
    if p.exists() and p.is_dir():
        # require subfolders for diffusers
        return str(p)
    return sd_arg  # treat as HF repo id


def _normalize_ckpt(sd_raw: dict):
    # unwrap pl checkpoint wrappers and remove 'module.' prefix
    if not isinstance(sd_raw, dict):
        raise RuntimeError("checkpoint is not a dict")
    if "state_dict" in sd_raw and isinstance(sd_raw["state_dict"], dict):
        sd = sd_raw["state_dict"]
    elif "model" in sd_raw and isinstance(sd_raw["model"], dict):
        sd = sd_raw["model"]
    else:
        sd = sd_raw
    new_sd = {}
    for k, v in sd.items():
        nk = k
        if nk.startswith("module."):
            nk = nk[len("module."):]
        new_sd[nk] = v
    return new_sd


def load_sd_checkpoint_to_vae_unet(ckpt_path: str, vae: torch.nn.Module, unet: torch.nn.Module,
                                   device: str = "cuda", dtype=torch.float16):
    ckpt_path = str(ckpt_path)
    if not Path(ckpt_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print(f"[Load] SD checkpoint: {ckpt_path} -> filtering for VAE / UNet")
    raw = torch.load(ckpt_path, map_location="cpu")
    sd = _normalize_ckpt(raw)

    vae_state = {}
    unet_state = {}
    other_keys = {}

    vae_keys = set(vae.state_dict().keys())
    unet_keys = set(unet.state_dict().keys())

    for k, v in sd.items():
        # common prefixes mapping
        if k.startswith("first_stage_model."):
            nk = k[len("first_stage_model."):]
            if nk in vae_keys:
                vae_state[nk] = v
                continue
        if k.startswith("vae."):
            nk = k[len("vae."):]
            if nk in vae_keys:
                vae_state[nk] = v
                continue
        if k.startswith("model.diffusion_model."):
            nk = k[len("model.diffusion_model."):]
            if nk in unet_keys:
                unet_state[nk] = v
                continue
        if k.startswith("diffusion_model."):
            nk = k[len("diffusion_model."):]
            if nk in unet_keys:
                unet_state[nk] = v
                continue
        if k.startswith("unet."):
            nk = k[len("unet."):]
            if nk in unet_keys:
                unet_state[nk] = v
                continue
        # fallback: direct key match
        if k in vae_keys:
            vae_state[k] = v
            continue
        if k in unet_keys:
            unet_state[k] = v
            continue
        other_keys[k] = v

    # load into modules with strict=False
    if vae_state:
        print(f"[Info] Loading {len(vae_state)} keys into VAE (strict=False)")
        missing, unexpected = vae.load_state_dict(vae_state, strict=False)
        print(f"  VAE: missing_keys={len(missing)}, unexpected_keys={len(unexpected)}")
    else:
        print("[Warn] No VAE keys matched the VAE state_dict. VAE unchanged.")

    if unet_state:
        print(f"[Info] Loading {len(unet_state)} keys into UNet (strict=False)")
        missing, unexpected = unet.load_state_dict(unet_state, strict=False)
        print(f"  UNet: missing_keys={len(missing)}, unexpected_keys={len(unexpected)}")
    else:
        print("[Warn] No UNet keys matched the UNet state_dict. UNet unchanged.")

    # Move & cast & freeze
    _dtype = dtype
    vae.to(device=device, dtype=_dtype)
    unet.to(device=device, dtype=_dtype)
    vae.eval()
    unet.eval()
    for p in vae.parameters():
        p.requires_grad = False
    for p in unet.parameters():
        p.requires_grad = False

    print(f"[Done] checkpoint applied. VAE/UNet moved to {device} dtype={_dtype} and frozen.")


def try_load_adapter_from_dir(unet: torch.nn.Module, adapter_dir: Union[str, Path]):
    adapter_dir = Path(adapter_dir)
    if not adapter_dir.exists():
        print(f"[Warn] adapter dir not found: {adapter_dir}")
        return
    # Prefer model-provided loader
    loader_found = False
    if hasattr(unet, "load_adapter_from_dir") and callable(getattr(unet, "load_adapter_from_dir")):
        try:
            unet.load_adapter_from_dir(str(adapter_dir))
            print(f"[Info] Adapter loaded via unet.load_adapter_from_dir({adapter_dir})")
            loader_found = True
        except Exception as e:
            print(f"[Warn] unet.load_adapter_from_dir failed: {e}")
    # try nested attributes common in ldm wrappers
    if not loader_found:
        for attr_path in [["model", "diffusion_model"], ["diffusion_model"], ["model"]]:
            obj = unet
            ok = True
            for a in attr_path:
                if hasattr(obj, a):
                    obj = getattr(obj, a)
                else:
                    ok = False
                    break
            if ok and hasattr(obj, "load_adapter_from_dir") and callable(getattr(obj, "load_adapter_from_dir")):
                try:
                    obj.load_adapter_from_dir(str(adapter_dir))
                    print(f"[Info] Adapter loaded via {'.'.join(attr_path)}.load_adapter_from_dir({adapter_dir})")
                    loader_found = True
                    break
                except Exception as e:
                    print(f"[Warn] nested load_adapter_from_dir failed: {e}")
    # fallback: manual .pt/.pth load with loose matching
    if not loader_found:
        candidates = list(adapter_dir.glob("*.pt")) + list(adapter_dir.glob("*.pth")) + list(adapter_dir.glob("*.bin"))
        if not candidates:
            print(f"[Warn] No adapter checkpoint files found in {adapter_dir}")
            return
        ckpt = torch.load(str(candidates[0]), map_location="cpu")
        sd = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        # remove module. prefix
        normalized = {}
        for k, v in (sd.items() if isinstance(sd, dict) else []):
            nk = k[len("module."):] if k.startswith("module.") else k
            normalized[nk] = v
        filtered = {}
        unet_keys = set(unet.state_dict().keys())
        for k, v in normalized.items():
            if k in unet_keys:
                filtered[k] = v
        if not filtered:
            print("[Warn] no matching keys found in adapter checkpoint for unet")
            return
        missing, unexpected = unet.load_state_dict(filtered, strict=False)
        print(f"[Info] Adapter checkpoint loaded (filtered). missing_keys={missing}, unexpected_keys={unexpected}")


def list_images(img_dir: Path) -> List[Path]:
    files = []
    for name in sorted(os.listdir(img_dir)):
        p = img_dir / name
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            files.append(p)
    return files


def _build_scheduler_from_cfg_params(cfg_params: dict):
    """
    根据 cfg 中的 linear_start/linear_end/timesteps 构建 DDPMScheduler。
    如果 cfg_params 为空则使用 v1.4 常用默认值。
    """
    linear_start = float(cfg_params.get("linear_start", 0.00085))
    linear_end = float(cfg_params.get("linear_end", 0.0120))
    timesteps = int(cfg_params.get("timesteps", 1000))
    return DDPMScheduler(
        beta_start=linear_start,
        beta_end=linear_end,
        beta_schedule="linear",
        num_train_timesteps=timesteps,
    )


def _load_clip_from_cond_cfg(cond_cfg, device, dtype):
    """
    从 cfg 中 cond_stage_config.params.version 指定的位置加载 CLIP tokenizer/text_encoder。
    返回 (tokenizer, text_encoder) 或抛错。
    """
    clip_version = None
    if cond_cfg and isinstance(cond_cfg, dict):
        clip_version = cond_cfg.get("params", {}).get("version") if isinstance(cond_cfg.get("params", {}), dict) else None

    if clip_version is None:
        raise RuntimeError("cond_stage_config.params.version not found in cfg; cannot load CLIP from config path")

    tokenizer = CLIPTokenizer.from_pretrained(clip_version)
    text_encoder = CLIPTextModel.from_pretrained(clip_version).to(DEVICE).eval()
    text_encoder = text_encoder.to(device=device, dtype=dtype)
    for p in text_encoder.parameters():
        p.requires_grad = False
    return tokenizer, text_encoder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img-dir", type=str, required=True, help="Folder of images (COCO train2017/val2017 etc.)")
    ap.add_argument("--out-dir", type=str, required=True, help="Where to save .pt feature files")
    ap.add_argument("--sd", type=str, required=True,
                    help="HF repo id OR local SD root containing: unet/, scheduler/, tokenizer/, text_encoder/, vae/ OR a single checkpoint file (.ckpt/.pth/.pt)")
    ap.add_argument("--cfg", type=str, default=None,
                    help="Optional: path to YAML config (e.g. unet-finetune-master/configs/.../attn2o-ffni.yaml). If provided, will instantiate unet/vae from it when possible.")
    ap.add_argument("--vae", type=str, default=None,
                    help="If provided, use this VAE subfolder from sd root or path")
    ap.add_argument("--image-size", type=int, default=512,
                    help="target shortest side for resize pipeline")
    ap.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "fp32"])
    ap.add_argument("--t-list", type=str, default="400,600",
                    help="timesteps to extract")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing .pt")
    ap.add_argument("--adapter-dir", type=str, default=None, help="Optional adapter dir to load into UNet")
    args = ap.parse_args()

    img_dir = Path(args.img_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sd_arg = args.sd
    sd_path = Path(sd_arg)
    sd_ckpt_file = None
    sd_root = None
    if sd_path.exists() and sd_path.is_file() and sd_path.suffix.lower() in {".ckpt", ".pth", ".pt"}:
        sd_ckpt_file = str(sd_path)
        sd_root = None
    else:
        sd_root = ensure_sd_root(sd_arg)

    # Try to load models from YAML cfg if provided
    models_from_cfg = {}
    if args.cfg is not None:
        try:
            models_from_cfg = load_models_from_yaml(args.cfg)
        except Exception as e:
            print(f"[Warn] load_models_from_yaml failed: {e}", file=sys.stderr)
            models_from_cfg = {}

    # VAE / UNET: instantiate from cfg if possible, else will use from_pretrained or checkpoint load
    if "vae" in models_from_cfg and models_from_cfg["vae"] is not None:
        vae = models_from_cfg["vae"]
        print("[Info] Using VAE instance from cfg")
    else:
        vae = None

    if "unet" in models_from_cfg and models_from_cfg["unet"] is not None:
        unet = models_from_cfg["unet"]
        print("[Info] Using UNet instance from cfg")
    else:
        unet = None

    # dtype
    dtype = torch.float16 if args.dtype == "fp16" else torch.float32

    # Load VAE/UNet:
    if sd_ckpt_file:
        # 优先：把 sd v1.4 checkpoint 的权重分发到 vae/unet
        if vae is None or unet is None:
            # 若 cfg 中没有实例化对象，则从 diffusers 类构造默认实例以接收权重
            if vae is None:
                vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae").to(DEVICE).eval()
            if unet is None:
                unet = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet").to(DEVICE).eval()

        try:
            load_sd_checkpoint_to_vae_unet(sd_ckpt_file, vae, unet, device=DEVICE, dtype=dtype)
            print(f"[Load] Applied checkpoint weights from {sd_ckpt_file} into VAE/UNet")
        except Exception as e:
            print(f"[Error] Failed to load checkpoint into VAE/UNet: {e}", file=sys.stderr)
            raise

        # Scheduler: build from cfg params (如果有 cfg 提供) 或使用 v1.4 默认 schedule
        cfg_params = (models_from_cfg.get("_cfg", {}).get("model", {}) or {}).get("params", {}) or {}
        scheduler = _build_scheduler_from_cfg_params(cfg_params)
        print("[Info] Scheduler built from cfg params / defaults for checkpoint-only mode")
    else:
        # sd_root mode: load from pretrained diffusers repo
        if vae is None:
            try:
                print("[Load] VAE:", args.vae if args.vae is not None else f"{sd_root}/vae")
                if args.vae is not None:
                    vae = AutoencoderKL.from_pretrained(args.vae).to(DEVICE).eval()
                else:
                    vae = AutoencoderKL.from_pretrained(sd_root, subfolder="vae").to(DEVICE).eval()
            except Exception as e:
                raise RuntimeError(f"Failed to load VAE from {sd_root or args.vae}: {e}")
        if unet is None:
            try:
                print("[Load] UNet:", sd_root, "(subfolder=unet)")
                unet = UNet2DConditionModel.from_pretrained(sd_root, subfolder="unet").to(DEVICE).eval()
            except Exception as e:
                raise RuntimeError(f"Failed to load UNet from {sd_root}: {e}")

        # Scheduler from sd_root
        scheduler = DDPMScheduler.from_pretrained(sd_root, subfolder="scheduler")
        print("[Load] Scheduler from sd_root")

    # Ensure models are on correct device/dtype and frozen
    vae.to(device=DEVICE, dtype=dtype).eval()
    unet.to(device=DEVICE, dtype=dtype).eval()
    for p in vae.parameters():
        p.requires_grad = False
    for p in unet.parameters():
        p.requires_grad = False

    # Tokenizer & TextEncoder: 优先使用 cfg 中 cond_stage_config.params.version，再回退到 sd_root 下的 tokenizer/text_encoder
    used_cfg = models_from_cfg.get("_cfg", {})
    cond_cfg = (used_cfg.get("model", {}) or {}).get("params", {}).get("cond_stage_config", None)
    tokenizer = None
    text_encoder = None
    try:
        if cond_cfg:
            tokenizer, text_encoder = _load_clip_from_cond_cfg(cond_cfg, DEVICE, dtype)
            print("[Load] CLIP loaded from cfg cond_stage_config.version")
        elif sd_root:
            print("[Load] Tokenizer & TextEncoder:", sd_root, "(subfolders=tokenizer,text_encoder)")
            tokenizer = CLIPTokenizer.from_pretrained(sd_root, subfolder="tokenizer")
            text_encoder = CLIPTextModel.from_pretrained(sd_root, subfolder="text_encoder").to(DEVICE).eval()
            text_encoder = text_encoder.to(device=DEVICE, dtype=dtype)
            for p in text_encoder.parameters():
                p.requires_grad = False
        else:
            raise RuntimeError("No cond_stage_config in cfg and no sd_root to load tokenizer/text_encoder from.")
    except Exception as e:
        raise RuntimeError(f"Failed to load CLIP tokenizer/text_encoder: {e}")

    # build unconditional embedding
    tokens = tokenizer([""], padding="max_length", max_length=tokenizer.model_max_length, return_tensors="pt")
    input_ids = tokens["input_ids"].to(DEVICE)
    with torch.no_grad():
        uncond_emb = text_encoder(input_ids)[0].to(dtype)

    # adapter loading (参考 sample.py)
    if args.adapter_dir:
        try:
            try_load_adapter_from_dir(unet, args.adapter_dir)
            print(f"[Info] Adapter loaded from ` {args.adapter_dir} `")
        except Exception as e:
            print(f"[Warn] Adapter load failed: {e}", file=sys.stderr)

    # parse t_list
    t_list = [int(x) for x in args.t_list.split(",") if x.strip() != ""]
    if len(t_list) == 0:
        raise ValueError("--t-list must contain at least one timestep")

    images = list_images(img_dir)
    if not images:
        raise RuntimeError(f"No images found in ` {img_dir} `")

    print(f"[Info] {len(images)} images found. Saving to: ` {out_dir} `")
    for img_path in tqdm(images, ncols=100):
        out_path = out_dir / (img_path.stem + ".pt")
        if out_path.exists() and not args.overwrite:
            continue
        try:
            x, (H0, W0), (Hc, Wc) = load_image_tensor(img_path, args.image_size)
            feats = extract_multiscale_latent_feats(vae, unet, scheduler, x, t_list, uncond_emb, dtype=dtype)
            p8, p16, p32 = feats["p8"], feats["p16"], feats["p32"]
            C8, Hp8, Wp8 = p8.shape
            C16, Hp16, Wp16 = p16.shape
            C32, Hp32, Wp32 = p32.shape
            payload = {
                "features": [p8, p16, p32],
                "order": ["p8", "p16", "p32"],
                "shapes": {
                    "p8":  (int(C8),  int(Hp8),  int(Wp8)),
                    "p16": (int(C16), int(Hp16), int(Wp16)),
                    "p32": (int(C32), int(Hp32), int(Wp32)),
                },
                "orig_size": (int(H0), int(W0)),
                "canonical_size": (int(Hc), int(Wc)),
                "grid_h8": (int(Hp8), int(Wp8)),
                "t_used": [int(t) for t in t_list],
                "sd_repo": str(args.sd),
                "vae_repo": str(args.vae) if args.vae is not None else (f"{sd_root}/vae" if sd_root else "checkpoint"),
                "dtype": "fp16" if dtype == torch.float16 else "fp32",
            }
            torch.save(payload, out_path)
        except Exception as e:
            print(f"[Error] {img_path}: {e}", file=sys.stderr)

    print("[Done]")


if __name__ == "__main__":
    main()