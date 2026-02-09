# ========================================
# Modified by Shoufa Chen
# ========================================
# Modified by Peize Sun, Rufeng Zhang
# Contact: {sunpeize, cxrfzhang}@foxmail.com
#
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import math
import random
import os
from typing import List
from collections import namedtuple

import torch
import torch.nn.functional as F
from torch import nn

from detectron2.layers import batched_nms
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, detector_postprocess
from detectron2.structures import Boxes, ImageList, Instances

from .loss import SetCriterionDynamicK, HungarianMatcherDynamicK
from .head import DynamicHead
from .util.box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh
from .util.misc import nested_tensor_from_tensor_list

__all__ = ["DiffusionDet"]
ModelPrediction = namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])


def exists(x): return x is not None


def default(val, d): return val if exists(val) else (d() if callable(d) else d)


def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))


def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


# ===== SD 残差门控 + 特征缓存加载器 =====
class ResidualGatedFuse(nn.Module):
    """
    y = x + alpha * phi(p)
    phi: 1x1 Conv -> GroupNorm -> SiLU
    alpha: 可训练标量，零初始化（起步等价 baseline）
    """

    def __init__(self, pin_channels, x_channels, num_groups=32):
        super().__init__()
        self.proj = nn.Conv2d(pin_channels, x_channels, kernel_size=1, bias=False)
        self.norm = nn.GroupNorm(num_groups=min(num_groups, x_channels), num_channels=x_channels)
        self.act = nn.SiLU(inplace=True)
        self.alpha = nn.Parameter(torch.zeros(1))  # 关键：零初始化，训练中逐步学注入强度

    def forward(self, x, p):  # x: C_k, p: P_k (已对齐到同 HxW)
        p = self.proj(p)
        p = self.norm(p)
        p = self.act(p)
        return x + self.alpha * p


class _SDFeatBank:
    """
    按 file_name 同名 .pt 读取离线 SD 特征（payload["features"]=[p8,p16,p32]）
    缺失文件会返回 None，占位后在融合时跳过该样本
    """

    def __init__(self, cache_dir, device, dtype="fp16"):
        self.cache_dir = cache_dir
        self.device = torch.device(device)
        self.dtype = torch.float16 if str(dtype).lower() == "fp16" else torch.float32

    def load_batch(self, file_names):
        feats = []
        for fn in file_names:
            stem = os.path.splitext(os.path.basename(fn))[0]
            path = os.path.join(self.cache_dir, stem + ".pt")
            if not os.path.exists(path):
                feats.append(None)
                continue
            payload = torch.load(path, map_location="cpu")
            try:
                # 兼容你导出脚本：payload["features"] = [p8,p16,p32]
                p8, p16, p32 = payload["features"]
            except Exception:
                feats.append(None)
                continue
            feats.append({
                "p8": p8.detach().to(self.device, dtype=self.dtype),
                "p16": p16.detach().to(self.device, dtype=self.dtype),
                "p32": p32.detach().to(self.device, dtype=self.dtype),
            })
        return feats


@META_ARCH_REGISTRY.register()
class DiffusionDet(nn.Module):
    """Implement DiffusionDet (+ optional CAL + optional SD 残差门控融合 v1)"""

    def __init__(self, cfg):
        super().__init__()
        self.device = torch.device(cfg.MODEL.DEVICE)

        # ---- Basic Config ----
        self.in_features = cfg.MODEL.ROI_HEADS.IN_FEATURES
        self.num_classes = cfg.MODEL.DiffusionDet.NUM_CLASSES
        self.num_proposals = cfg.MODEL.DiffusionDet.NUM_PROPOSALS
        self.hidden_dim = cfg.MODEL.DiffusionDet.HIDDEN_DIM
        self.num_heads = cfg.MODEL.DiffusionDet.NUM_HEADS
        self.cal_on = getattr(cfg.MODEL.DiffusionDet, "CAL_ON", False)

        # ---- Backbone ----
        self.backbone = build_backbone(cfg)
        self.size_divisibility = self.backbone.size_divisibility

        # ---- Diffusion Schedule ----
        timesteps = 1000
        sampling_timesteps = cfg.MODEL.DiffusionDet.SAMPLE_STEP
        self.objective = 'pred_x0'
        betas = cosine_beta_schedule(timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)
        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        self.sampling_timesteps = default(sampling_timesteps, timesteps)
        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = 1.
        self.self_condition = False
        self.scale = cfg.MODEL.DiffusionDet.SNR_SCALE
        self.box_renewal = True
        self.use_ensemble = True

        # Register buffers
        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)
        self.register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min=1e-20)))
        self.register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
                             (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # ---- Dynamic Head ----
        self.head = DynamicHead(cfg=cfg, roi_input_shape=self.backbone.output_shape())

        # ---- Loss Config ----
        class_weight = cfg.MODEL.DiffusionDet.CLASS_WEIGHT
        giou_weight = cfg.MODEL.DiffusionDet.GIOU_WEIGHT
        l1_weight = cfg.MODEL.DiffusionDet.L1_WEIGHT
        no_object_weight = cfg.MODEL.DiffusionDet.NO_OBJECT_WEIGHT
        self.deep_supervision = cfg.MODEL.DiffusionDet.DEEP_SUPERVISION
        self.use_focal = cfg.MODEL.DiffusionDet.USE_FOCAL
        self.use_fed_loss = cfg.MODEL.DiffusionDet.USE_FED_LOSS
        self.use_nms = cfg.MODEL.DiffusionDet.USE_NMS

        matcher = HungarianMatcherDynamicK(cfg, cost_class=class_weight, cost_bbox=l1_weight,
                                           cost_giou=giou_weight, use_focal=self.use_focal)
        weight_dict = {"loss_ce": class_weight, "loss_bbox": l1_weight, "loss_giou": giou_weight}

        if self.deep_supervision:
            aux_weight_dict = {}
            for i in range(self.num_heads - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        losses = ["labels", "boxes"]

        self.criterion = SetCriterionDynamicK(
            cfg=cfg, num_classes=self.num_classes, matcher=matcher, weight_dict=weight_dict,
            eos_coef=no_object_weight, losses=["labels", "boxes"], use_focal=self.use_focal
        )

        # ---- Normalization ----
        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(3, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(3, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std

        # ================== SD 残差门控融合（v1）设置 ==================
        # 简易 cfg 项：你也可以转到 config.py 正式注册
        self.sdfuse_enable = getattr(cfg.MODEL, "SDFUSE_ENABLE", True)  # 开关
        self.sdfuse_cache = getattr(cfg.MODEL, "SDFUSE_CACHE_DIR", "/mnt/dc_cache")
        self.sdfuse_dtype = getattr(cfg.MODEL, "SDFUSE_DTYPE", "fp16")
        # SD v2.1 base 默认通道
        self.sdfuse_p_ch = getattr(cfg.MODEL, "SDFUSE_P_CH", {"p8": 320, "p16": 640, "p32": 1280})

        # 从 backbone 输出描述中找到 stride=8/16/32
        out_shapes = self.backbone.output_shape()  # dict: name -> ShapeSpec(channels, stride, ...)
        by_stride = {}
        for name, spec in out_shapes.items():
            by_stride[spec.stride] = (name, spec.channels)
        self._k8 = by_stride.get(8, None)  # (name, C3_channels)
        self._k16 = by_stride.get(16, None)  # (name, C4_channels)
        self._k32 = by_stride.get(32, None)  # (name, C5_channels)

        if self.sdfuse_enable and (self._k8 or self._k16 or self._k32):
            self.sd_bank = _SDFeatBank(self.sdfuse_cache, device=self.device, dtype=self.sdfuse_dtype)
            if self._k8 is not None:
                _, c3 = self._k8
                self.fuse_c3 = ResidualGatedFuse(self.sdfuse_p_ch["p8"], c3)
            else:
                self.fuse_c3 = None
            if self._k16 is not None:
                _, c4 = self._k16
                self.fuse_c4 = ResidualGatedFuse(self.sdfuse_p_ch["p16"], c4)
            else:
                self.fuse_c4 = None
            if self._k32 is not None:
                _, c5 = self._k32
                self.fuse_c5 = ResidualGatedFuse(self.sdfuse_p_ch["p32"], c5)
            else:
                self.fuse_c5 = None
        else:
            self.sd_bank = None
            self.fuse_c3 = self.fuse_c4 = self.fuse_c5 = None

        self.to(self.device)

    # =========================================================
    # Core Functions
    # =========================================================
    def predict_noise_from_start(self, x_t, t, x0):
        return ((extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) /
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape))

    def model_predictions(self, feats, images_whwh, x, t, x_self_cond=None, clip_x_start=False):
        x_boxes = torch.clamp(x, -self.scale, self.scale)
        x_boxes = ((x_boxes / self.scale) + 1) / 2
        x_boxes = box_cxcywh_to_xyxy(x_boxes) * images_whwh[:, None, :]
        outputs_class, outputs_coord, _ = self.head(feats, x_boxes, t, None, proposal_gt_cls=None)
        x_start = outputs_coord[-1] / images_whwh[:, None, :]
        x_start = (box_xyxy_to_cxcywh(x_start) * 2 - 1) * self.scale
        pred_noise = self.predict_noise_from_start(x, t, torch.clamp(x_start, -self.scale, self.scale))
        return ModelPrediction(pred_noise, x_start), outputs_class, outputs_coord

    @torch.no_grad()
    def ddim_sample(self, batched_inputs, backbone_feats, images_whwh, images, clip_denoised=True, do_postprocess=True):
        batch = images_whwh.shape[0]
        shape = (batch, self.num_proposals, 4)
        total_timesteps, sampling_timesteps, eta, objective = self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective

        # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        img = torch.randn(shape, device=self.device)

        ensemble_score, ensemble_label, ensemble_coord = [], [], []
        x_start = None
        for time, time_next in time_pairs:
            time_cond = torch.full((batch,), time, device=self.device, dtype=torch.long)
            self_cond = x_start if self.self_condition else None

            preds, outputs_class, outputs_coord = self.model_predictions(backbone_feats, images_whwh, img, time_cond,
                                                                         self_cond, clip_x_start=clip_denoised)
            pred_noise, x_start = preds.pred_noise, preds.pred_x_start

            if self.box_renewal:  # filter
                score_per_image, box_per_image = outputs_class[-1][0], outputs_coord[-1][0]
                threshold = 0.5
                score_per_image = torch.sigmoid(score_per_image)
                value, _ = torch.max(score_per_image, -1, keepdim=False)
                keep_idx = value > threshold
                num_remain = torch.sum(keep_idx)

                pred_noise = pred_noise[:, keep_idx, :]
                x_start = x_start[:, keep_idx, :]
                img = img[:, keep_idx, :]
            if time_next < 0:
                img = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(img)

            img = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise

            if self.box_renewal:  # filter
                # replenish with randn boxes
                img = torch.cat((img, torch.randn(1, self.num_proposals - num_remain, 4, device=img.device)), dim=1)
            if self.use_ensemble and self.sampling_timesteps > 1:
                box_pred_per_image, scores_per_image, labels_per_image = self.inference(outputs_class[-1],
                                                                                        outputs_coord[-1],
                                                                                        images.image_sizes)
                ensemble_score.append(scores_per_image)
                ensemble_label.append(labels_per_image)
                ensemble_coord.append(box_pred_per_image)

        if self.use_ensemble and self.sampling_timesteps > 1:
            box_pred_per_image = torch.cat(ensemble_coord, dim=0)
            scores_per_image = torch.cat(ensemble_score, dim=0)
            labels_per_image = torch.cat(ensemble_label, dim=0)
            if self.use_nms:
                keep = batched_nms(box_pred_per_image, scores_per_image, labels_per_image, 0.5)
                box_pred_per_image = box_pred_per_image[keep]
                scores_per_image = scores_per_image[keep]
                labels_per_image = labels_per_image[keep]

            result = Instances(images.image_sizes[0])
            result.pred_boxes = Boxes(box_pred_per_image)
            result.scores = scores_per_image
            result.pred_classes = labels_per_image
            results = [result]
        else:
            output = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
            box_cls = output["pred_logits"]
            box_pred = output["pred_boxes"]
            results = self.inference(box_cls, box_pred, images.image_sizes)
        if do_postprocess:
            processed_results = []
            for results_per_image, input_per_image, image_size in zip(results, batched_inputs, images.image_sizes):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                r = detector_postprocess(results_per_image, height, width)
                processed_results.append({"instances": r})
            return processed_results

    def q_sample(self, x_start, t, noise=None):
        if noise is None: noise = torch.randn_like(x_start)
        return (extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise)

    # ======================= SD 残差门控融合（前-FPN）=======================
    def _fuse_sd_into_src(self, src, batched_inputs):
        """
        src: backbone(images.tensor) 的输出 dict (name -> [B,C,H,W])
        将 SD 的 p8/p16/p32 残差门控注入到 stride=8/16/32 对应的特征层
        """
        if (not self.sdfuse_enable) or (self.sd_bank is None):
            return src

        file_names = [x["file_name"] for x in batched_inputs]
        feats = self.sd_bank.load_batch(file_names)
        if all(f is None for f in feats):
            return src

        B = len(batched_inputs)

        def _stack_or_none(key):
            arr = [f[key] for f in feats if f is not None]
            if len(arr) == 0:
                return None
            stacked = []
            idx = 0
            for f in feats:
                if f is None:
                    stacked.append(None)
                else:
                    stacked.append(arr[idx]);
                    idx += 1
            return stacked

        P8_list = _stack_or_none("p8")
        P16_list = _stack_or_none("p16")
        P32_list = _stack_or_none("p32")

        # stride=8 -> C3
        if (self._k8 is not None) and (self.fuse_c3 is not None) and (P8_list is not None):
            name8, _ = self._k8
            C3 = src[name8]
            target_hw = C3.shape[-2:]
            P8 = torch.stack([p for p in P8_list if p is not None], dim=0)
            P8 = F.interpolate(P8, size=target_hw, mode="bilinear", align_corners=False)
            bptr = 0;
            fused = []
            for i in range(B):
                if P8_list[i] is None:
                    fused.append(C3[i:i + 1])
                else:
                    fused.append(self.fuse_c3(C3[i:i + 1], P8[bptr:bptr + 1]));
                    bptr += 1
            src[name8] = torch.cat(fused, dim=0)

        # stride=16 -> C4
        if (self._k16 is not None) and (self.fuse_c4 is not None) and (P16_list is not None):
            name16, _ = self._k16
            C4 = src[name16]
            target_hw = C4.shape[-2:]
            P16 = torch.stack([p for p in P16_list if p is not None], dim=0)
            P16 = F.interpolate(P16, size=target_hw, mode="bilinear", align_corners=False)
            bptr = 0;
            fused = []
            for i in range(B):
                if P16_list[i] is None:
                    fused.append(C4[i:i + 1])
                else:
                    fused.append(self.fuse_c4(C4[i:i + 1], P16[bptr:bptr + 1]));
                    bptr += 1
            src[name16] = torch.cat(fused, dim=0)

        # stride=32 -> C5
        if (self._k32 is not None) and (self.fuse_c5 is not None) and (P32_list is not None):
            name32, _ = self._k32
            C5 = src[name32]
            target_hw = C5.shape[-2:]
            P32 = torch.stack([p for p in P32_list if p is not None], dim=0)
            P32 = F.interpolate(P32, size=target_hw, mode="bilinear", align_corners=False)
            bptr = 0;
            fused = []
            for i in range(B):
                if P32_list[i] is None:
                    fused.append(C5[i:i + 1])
                else:
                    fused.append(self.fuse_c5(C5[i:i + 1], P32[bptr:bptr + 1]));
                    bptr += 1
            src[name32] = torch.cat(fused, dim=0)

        return src

    def forward(self, batched_inputs, do_postprocess=True):
        images, images_whwh = self.preprocess_image(batched_inputs)
        if isinstance(images, (list, torch.Tensor)):
            images = nested_tensor_from_tensor_list(images)

        # === Backbone ===
        src_raw = self.backbone(images.tensor)  # 保留原始特征（给 CAL 用）

        # === 主分支：SD 残差门控融合（前-FPN） ===
        # 注意：传入 dict(src_raw)（浅拷贝），_fuse 会在副本上替换 tensor，不会改动 src_raw
        src_fused = self._fuse_sd_into_src(dict(src_raw), batched_inputs)

        # 主分支特征（融合后）
        features_main = [src_fused[f] for f in self.in_features]

        # === Inference ===
        if not self.training:
            return self.ddim_sample(batched_inputs, features_main, images_whwh, images)

        # === Training ===
        gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        targets, x_boxes, noises, t = self.prepare_targets(gt_instances)
        t = t.squeeze(-1)
        x_boxes = x_boxes * images_whwh[:, None, :]

        # --------------------------------------------------------------------
        # 主分支（融合特征）baseline forward
        # --------------------------------------------------------------------
        outputs_class, outputs_coord, _ = self.head(features_main, x_boxes, t, None, proposal_gt_cls=None)
        output = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}

        if self.deep_supervision:
            output['aux_outputs'] = [
                {'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])
            ]

        # --- 标准损失 ---
        loss_dict = self.criterion(output, targets)

        # --------------------------------------------------------------------
        # CAL forward  —— 用“未融合”的原特征
        # --------------------------------------------------------------------
        if self.cal_on:
            # 建议：不让 CAL 回传到 backbone（更稳），如需联训可去掉 .detach()
            features_cal = [src_fused[f] for f in self.in_features]

            with torch.no_grad():
                indices, _ = self.criterion.matcher(
                    {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}, targets
                )
                N, P = outputs_class[-1].shape[:2]
                proposal_gt_cls = torch.full((N, P), -1, dtype=torch.long, device=self.device)
                for b, (Ipred, Igt) in enumerate(indices):
                    pred_ids = (
                        torch.nonzero(Ipred, as_tuple=False).squeeze(1)
                        if Ipred.dtype == torch.bool else Ipred
                    )
                    if pred_ids.numel() == 0:
                        continue
                    gt_cls = targets[b]["labels"][Igt]
                    proposal_gt_cls[b].index_copy_(0, pred_ids, gt_cls)

            # 用原特征跑 CAL 分支
            _, _, cal_extra = self.head(features_cal, x_boxes, t, None, proposal_gt_cls=proposal_gt_cls)

            # --- 合并所有层的 CAL loss（保持你原逻辑） ---
            if isinstance(cal_extra, dict):
                for k, v in cal_extra.items():
                    if v is None:
                        continue
                    if torch.is_tensor(v):
                        loss_dict[k] = torch.nan_to_num(v, nan=0.0)
            elif cal_extra is not None and cal_extra.get("cal_loss") is not None:
                loss_dict["loss_cal"] = cal_extra["cal_loss"]

            cal_keys = [k for k in loss_dict.keys() if k.startswith("loss_cal_")]
            if cal_keys:
                loss_dict["loss_cal"] = sum(loss_dict[k] for k in cal_keys) / len(cal_keys)

        # --------------------------------------------------------------------
        # 按权重表缩放（CAL 项已在 head 里乘 λ）
        # --------------------------------------------------------------------
        for k in list(loss_dict.keys()):
            if k in self.criterion.weight_dict:
                loss_dict[k] *= self.criterion.weight_dict[k]

        return loss_dict

    # =========================================================
    # Other Utility Functions
    # =========================================================
    def prepare_diffusion_repeat(self, gt_boxes):
        """
        :param gt_boxes: (cx, cy, w, h), normalized
        :param num_proposals:
        """
        t = torch.randint(0, self.num_timesteps, (1,), device=self.device).long()
        noise = torch.randn(self.num_proposals, 4, device=self.device)

        num_gt = gt_boxes.shape[0]
        if not num_gt:  # generate fake gt boxes if empty gt boxes
            gt_boxes = torch.as_tensor([[0.5, 0.5, 1., 1.]], dtype=torch.float, device=self.device)
            num_gt = 1

        num_repeat = self.num_proposals // num_gt  # number of repeat except the last gt box in one image
        repeat_tensor = [num_repeat] * (num_gt - self.num_proposals % num_gt) + [num_repeat + 1] * (
                self.num_proposals % num_gt)
        assert sum(repeat_tensor) == self.num_proposals
        random.shuffle(repeat_tensor)
        repeat_tensor = torch.tensor(repeat_tensor, device=self.device)

        gt_boxes = (gt_boxes * 2. - 1.) * self.scale
        x_start = torch.repeat_interleave(gt_boxes, repeat_tensor, dim=0)

        # noise sample
        x = self.q_sample(x_start=x_start, t=t, noise=noise)

        x = torch.clamp(x, min=-1 * self.scale, max=self.scale)
        x = ((x / self.scale) + 1) / 2.

        diff_boxes = box_cxcywh_to_xyxy(x)

        return diff_boxes, noise, t

    def prepare_diffusion_concat(self, gt_boxes):
        """
        :param gt_boxes: (cx, cy, w, h), normalized
        :param num_proposals:
        """
        t = torch.randint(0, self.num_timesteps, (1,), device=self.device).long()
        noise = torch.randn(self.num_proposals, 4, device=self.device)

        num_gt = gt_boxes.shape[0]
        if not num_gt:  # generate fake gt boxes if empty gt boxes
            gt_boxes = torch.as_tensor([[0.5, 0.5, 1., 1.]], dtype=torch.float, device=self.device)
            num_gt = 1

        if num_gt < self.num_proposals:
            box_placeholder = torch.randn(self.num_proposals - num_gt, 4,
                                          device=self.device) / 6. + 0.5  # 3sigma = 1/2 --> sigma: 1/6
            box_placeholder[:, 2:] = torch.clip(box_placeholder[:, 2:], min=1e-4)
            x_start = torch.cat((gt_boxes, box_placeholder), dim=0)
        elif num_gt > self.num_proposals:
            select_mask = [True] * self.num_proposals + [False] * (num_gt - self.num_proposals)
            random.shuffle(select_mask)
            x_start = gt_boxes[select_mask]
        else:
            x_start = gt_boxes

        x_start = (x_start * 2. - 1.) * self.scale

        # noise sample
        x = self.q_sample(x_start=x_start, t=t, noise=noise)

        x = torch.clamp(x, min=-1 * self.scale, max=self.scale)
        x = ((x / self.scale) + 1) / 2.

        diff_boxes = box_cxcywh_to_xyxy(x)

        return diff_boxes, noise, t

    def prepare_targets(self, targets):
        new_targets, diff_boxes, noises, ts = [], [], [], []
        for tgt in targets:
            h, w = tgt.image_size
            image_size_xyxy = torch.tensor([w, h, w, h], device=self.device)
            gt_cls = tgt.gt_classes
            gt_boxes = box_xyxy_to_cxcywh(tgt.gt_boxes.tensor / image_size_xyxy)
            d_boxes, d_noise, d_t = self.prepare_diffusion_concat(gt_boxes)
            diff_boxes.append(d_boxes)
            noises.append(d_noise)
            ts.append(d_t)
            new_targets.append({
                "labels": gt_cls.to(self.device),
                "boxes": gt_boxes.to(self.device),
                "boxes_xyxy": tgt.gt_boxes.tensor.to(self.device),
                "image_size_xyxy": image_size_xyxy.to(self.device),
                "image_size_xyxy_tgt": image_size_xyxy.unsqueeze(0).repeat(len(gt_boxes), 1),
                "area": tgt.gt_boxes.area().to(self.device),
            })
        return new_targets, torch.stack(diff_boxes), torch.stack(noises), torch.stack(ts)

    def inference(self, box_cls, box_pred, image_sizes):
        """
        Arguments:
            box_cls (Tensor): tensor of shape (batch_size, num_proposals, K).
                The tensor predicts the classification probability for each proposal.
            box_pred (Tensor): tensors of shape (batch_size, num_proposals, 4).
                The tensor predicts 4-vector (x,y,w,h) box
                regression values for every proposal
            image_sizes (List[torch.Size]): the input image sizes

        Returns:
            results (List[Instances]): a list of #images elements.
        """
        assert len(box_cls) == len(image_sizes)
        results = []

        if self.use_focal or self.use_fed_loss:
            scores = torch.sigmoid(box_cls)
            labels = torch.arange(self.num_classes, device=self.device). \
                unsqueeze(0).repeat(self.num_proposals, 1).flatten(0, 1)

            for i, (scores_per_image, box_pred_per_image, image_size) in enumerate(zip(
                    scores, box_pred, image_sizes
            )):
                result = Instances(image_size)
                scores_per_image, topk_indices = scores_per_image.flatten(0, 1).topk(self.num_proposals, sorted=False)
                labels_per_image = labels[topk_indices]
                box_pred_per_image = box_pred_per_image.view(-1, 1, 4).repeat(1, self.num_classes, 1).view(-1, 4)
                box_pred_per_image = box_pred_per_image[topk_indices]

                if self.use_ensemble and self.sampling_timesteps > 1:
                    return box_pred_per_image, scores_per_image, labels_per_image

                if self.use_nms:
                    keep = batched_nms(box_pred_per_image, scores_per_image, labels_per_image, 0.5)
                    box_pred_per_image = box_pred_per_image[keep]
                    scores_per_image = scores_per_image[keep]
                    labels_per_image = labels_per_image[keep]

                result.pred_boxes = Boxes(box_pred_per_image)
                result.scores = scores_per_image
                result.pred_classes = labels_per_image
                results.append(result)

        else:
            # For each box we assign the best class or the second best if the best on is `no_object`.
            scores, labels = F.softmax(box_cls, dim=-1)[:, :, :-1].max(-1)

            for i, (scores_per_image, labels_per_image, box_pred_per_image, image_size) in enumerate(zip(
                    scores, labels, box_pred, image_sizes
            )):
                if self.use_ensemble and self.sampling_timesteps > 1:
                    return box_pred_per_image, scores_per_image, labels_per_image

                if self.use_nms:
                    keep = batched_nms(box_pred_per_image, scores_per_image, labels_per_image, 0.5)
                    box_pred_per_image = box_pred_per_image[keep]
                    scores_per_image = scores_per_image[keep]
                    labels_per_image = labels_per_image[keep]
                result = Instances(image_size)
                result.pred_boxes = Boxes(box_pred_per_image)
                result.scores = scores_per_image
                result.pred_classes = labels_per_image
                results.append(result)

        return results

    def preprocess_image(self, batched_inputs):
        images = [self.normalizer(x["image"].to(self.device)) for x in batched_inputs]
        images = ImageList.from_tensors(images, self.size_divisibility)
        images_whwh = torch.stack([
            torch.tensor([x["image"].shape[-1], x["image"].shape[-2],
                          x["image"].shape[-1], x["image"].shape[-2]],
                         dtype=torch.float32, device=self.device)
            for x in batched_inputs
        ])
        return images, images_whwh
