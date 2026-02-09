# ========================================
# Modified by Shoufa Chen (with CAL integration)
# ========================================
# Modified by Peize Sun, Rufeng Zhang
# Contact: {sunpeize, cxrfzhang}@foxmail.com
#
# Copyright (c) Facebook, Inc. and its affiliates.
# All Rights Reserved
"""
DiffusionDet Transformer head with optional Counterfactual Attention Learning (CAL).
"""
import copy
import math
import numpy as np
import torch
from torch import nn, Tensor
import torch.nn.functional as F

from detectron2.modeling.poolers import ROIPooler
from detectron2.structures import Boxes
from diffusiondet.util import box_ops


_DEFAULT_SCALE_CLAMP = math.log(100000.0 / 16)


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class GaussianFourierProjection(nn.Module):
    """Gaussian random features for encoding time steps."""
    def __init__(self, embed_dim, scale=30.):
        super().__init__()
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)

    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class Dense(nn.Module):
    """A fully connected layer that reshapes outputs to feature maps."""
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.dense = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.dense(x)


class DynamicHead(nn.Module):
    """
    Stack of RCNNHead with time conditioning and optional CAL loss on the last head.
    """
    def __init__(self, cfg, roi_input_shape):
        super().__init__()

        # Build RoI.
        box_pooler = self._init_box_pooler(cfg, roi_input_shape)
        self.box_pooler = box_pooler

        # Build heads.
        num_classes = cfg.MODEL.DiffusionDet.NUM_CLASSES
        d_model = cfg.MODEL.DiffusionDet.HIDDEN_DIM
        dim_feedforward = cfg.MODEL.DiffusionDet.DIM_FEEDFORWARD
        nhead = cfg.MODEL.DiffusionDet.NHEADS
        dropout = cfg.MODEL.DiffusionDet.DROPOUT
        activation = cfg.MODEL.DiffusionDet.ACTIVATION
        num_heads = cfg.MODEL.DiffusionDet.NUM_HEADS
        rcnn_head = RCNNHead(cfg, d_model, num_classes, dim_feedforward, nhead, dropout, activation)
        self.head_series = _get_clones(rcnn_head, num_heads)
        self.num_heads = num_heads
        self.return_intermediate = cfg.MODEL.DiffusionDet.DEEP_SUPERVISION

        # Gaussian random feature embedding layer for time
        self.d_model = d_model
        time_dim = d_model * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(d_model),
            nn.Linear(d_model, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        # Init parameters.
        self.use_focal = cfg.MODEL.DiffusionDet.USE_FOCAL
        self.use_fed_loss = cfg.MODEL.DiffusionDet.USE_FED_LOSS
        self.num_classes = num_classes
        if self.use_focal or self.use_fed_loss:
            prior_prob = cfg.MODEL.DiffusionDet.PRIOR_PROB
            self.bias_value = -math.log((1 - prior_prob) / prior_prob)
        self._reset_parameters()

        # === CAL configs (for orchestration; real logic in RCNNHead) ===
        self.cal_on    = getattr(cfg.MODEL.DiffusionDet, "CAL_ON", False)
        self.cal_types = tuple(getattr(cfg.MODEL.DiffusionDet, "CAL_TYPES", ("random", "uniform", "shuffle")))
        self.cal_prob  = float(getattr(cfg.MODEL.DiffusionDet, "CAL_PROB", 1.0))

    def _reset_parameters(self):
        # init all parameters.
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

            # initialize the bias for focal loss and fed loss.
            if self.use_focal or self.use_fed_loss:
                if p.shape[-1] == self.num_classes or p.shape[-1] == self.num_classes + 1:
                    nn.init.constant_(p, self.bias_value)

    @staticmethod
    def _init_box_pooler(cfg, input_shape):
        in_features = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales = tuple(1.0 / input_shape[k].stride for k in in_features)
        sampling_ratio = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE

        in_channels = [input_shape[f].channels for f in in_features]
        assert len(set(in_channels)) == 1, in_channels

        box_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        return box_pooler

    @torch.no_grad()
    def _choose_cf_type(self):
        if len(self.cal_types) == 0:
            return "random"
        idx = np.random.randint(0, len(self.cal_types))
        return self.cal_types[idx]

    def forward(self, features, init_bboxes, t, init_features, proposal_gt_cls=None):
        """
        Multi-layer CAL version:
          - Each head runs both factual and counterfactual branches.
          - Only the last head allows CAL gradients to backpropagate.
          - Earlier heads compute CAL loss in detached mode (monitoring only).
        """
        time = self.time_mlp(t)

        inter_class_logits, inter_pred_bboxes = [], []
        cal_losses = {}   # <-- record CAL loss for each head

        bs = len(features[0])
        bboxes = init_bboxes
        proposal_features = None if init_features is None else init_features[None].repeat(1, bs, 1)

        for head_idx, rcnn_head in enumerate(self.head_series):
            is_last = (head_idx == self.num_heads - 1)

            # =====================================================
            # factual forward (normal detection branch)
            # =====================================================
            class_logits, pred_bboxes, proposal_features = rcnn_head(
                features, bboxes, proposal_features, self.box_pooler, time,
                use_cf=False, cf_type="random"
            )

            if self.return_intermediate:
                inter_class_logits.append(class_logits)
                inter_pred_bboxes.append(pred_bboxes)

            # =====================================================
            # counterfactual forward (CAL branch)
            # =====================================================
            if (
                self.training
                and self.cal_on
                and (proposal_gt_cls is not None)
                and (np.random.rand() < self.cal_prob)
            ):
                cf_type = self._choose_cf_type()
                class_logits_cf, pred_bboxes_cf, _ = rcnn_head(
                    features, bboxes, proposal_features, self.box_pooler, time,
                    use_cf=True, cf_type=cf_type
                )

                mask = proposal_gt_cls.ge(0)
                if mask.any():
                    # === classification hinge loss ===
                    idx_b = torch.arange(proposal_gt_cls.size(0), device=bboxes.device)[:, None].expand_as(proposal_gt_cls)[mask]
                    idx_p = torch.arange(proposal_gt_cls.size(1), device=bboxes.device)[None, :].expand_as(proposal_gt_cls)[mask]
                    idx_c = proposal_gt_cls[mask]

                    s_f  = class_logits[idx_b, idx_p, idx_c]
                    s_cf = class_logits_cf[idx_b, idx_p, idx_c]

                    margin  = rcnn_head.cal_margin
                    lam_cls = rcnn_head.cal_lambda
                    loss_cls_cf = lam_cls * F.relu(margin - (s_f - s_cf)).mean()

                    # === regression consistency ===
                    lam_reg  = getattr(rcnn_head, "cal_lambda_reg", 0.3)
                    use_giou = getattr(rcnn_head, "cal_use_giou", True)

                    mask_flat = mask.view(-1)
                    pb_f  = pred_bboxes.reshape(-1, 4)[mask_flat]
                    pb_cf = pred_bboxes_cf.reshape(-1, 4)[mask_flat]

                    if pb_f.numel() > 0:
                        def sanitize_xyxy(x):
                            x = torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4)
                            x1y1 = torch.minimum(x[:, :2], x[:, 2:])
                            x2y2 = torch.maximum(x[:, :2], x[:, 2:])
                            return torch.cat([x1y1, x2y2], dim=-1)

                        pb_f  = sanitize_xyxy(pb_f)
                        pb_cf = sanitize_xyxy(pb_cf)
                        pb_f  = pb_f.detach()
                        pb_cf = pb_cf.detach()

                        # — only last head keeps CAL gradients —
                        if not is_last:
                            s_f  = s_f.detach()
                            s_cf = s_cf.detach()


                        if use_giou:
                            iou = box_ops.generalized_box_iou(pb_f, pb_cf)
                            iou = torch.nan_to_num(iou, nan=0.0)
                            loss_reg_cf = (1.0 - iou.diag().mean()) if iou.ndim == 2 else (1.0 - iou.mean())
                        else:
                            diff = torch.nan_to_num(pb_f - pb_cf, nan=0.0, posinf=1e4, neginf=-1e4)
                            loss_reg_cf = diff.abs().mean()
                    else:
                        loss_reg_cf = pred_bboxes.sum() * 0.0

                    # === combine classification + regression CAL loss ===
                    this_cal = loss_cls_cf + lam_reg * loss_reg_cf
                    cal_losses[f"loss_cal_{head_idx}"] = this_cal

            # =====================================================
            # detach proposals between heads (DiffusionDet default)
            # =====================================================
            bboxes = pred_bboxes.detach()

        # ==========================================================
        # collect and return all outputs
        # ==========================================================
        if self.return_intermediate:
            return (
                torch.stack(inter_class_logits),
                torch.stack(inter_pred_bboxes),
                cal_losses,
            )

        return class_logits[None], pred_bboxes[None], cal_losses




class RCNNHead(nn.Module):
    """
    One refinement head block (self-attn + dynamic conv + FFN) with optional CAL gating.
    """
    def __init__(self, cfg, d_model, num_classes, dim_feedforward=2048, nhead=8, dropout=0.1, activation="relu",
                 scale_clamp: float = _DEFAULT_SCALE_CLAMP, bbox_weights=(2.0, 2.0, 1.0, 1.0)):
        super().__init__()

        self.d_model = d_model

        # dynamic.
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.inst_interact = DynamicConv(cfg)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

        # block time mlp
        self.block_time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(d_model * 4, d_model * 2))

        # cls.
        num_cls = cfg.MODEL.DiffusionDet.NUM_CLS
        cls_module = list()
        for _ in range(num_cls):
            cls_module.append(nn.Linear(d_model, d_model, False))
            cls_module.append(nn.LayerNorm(d_model))
            cls_module.append(nn.ReLU(inplace=True))
        self.cls_module = nn.ModuleList(cls_module)

        # reg.
        num_reg = cfg.MODEL.DiffusionDet.NUM_REG
        reg_module = list()
        for _ in range(num_reg):
            reg_module.append(nn.Linear(d_model, d_model, False))
            reg_module.append(nn.LayerNorm(d_model))
            reg_module.append(nn.ReLU(inplace=True))
        self.reg_module = nn.ModuleList(reg_module)

        # pred.
        self.use_focal = cfg.MODEL.DiffusionDet.USE_FOCAL
        self.use_fed_loss = cfg.MODEL.DiffusionDet.USE_FED_LOSS
        if self.use_focal or self.use_fed_loss:
            self.class_logits = nn.Linear(d_model, num_classes)
        else:
            self.class_logits = nn.Linear(d_model, num_classes + 1)
        self.bboxes_delta = nn.Linear(d_model, 4)
        self.scale_clamp = scale_clamp
        self.bbox_weights = bbox_weights

        # === CAL configs & gating ===
        self.cal_on        = getattr(cfg.MODEL.DiffusionDet, "CAL_ON", False)
        self.cal_types     = tuple(getattr(cfg.MODEL.DiffusionDet, "CAL_TYPES", ("random", "uniform", "shuffle")))
        self.cal_margin    = float(getattr(cfg.MODEL.DiffusionDet, "CAL_MARGIN", 0.0))
        self.cal_lambda    = float(getattr(cfg.MODEL.DiffusionDet, "CAL_LAMBDA", 0.3))
        self.cal_prob      = float(getattr(cfg.MODEL.DiffusionDet, "CAL_PROB", 1.0))
        self.cal_detach_cf = bool(getattr(cfg.MODEL.DiffusionDet, "CAL_DETACH_CF", True))

        # channel-wise attention gate on proposal features [N,P,C] -> (0,1)
        self.cal_gate = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model), nn.Sigmoid()
        )

    def forward(self, features, bboxes, pro_features, pooler, time_emb, use_cf: bool=False, cf_type: str="random"):
        """
        :param features: list of FPN features
        :param bboxes: (N, P, 4) absolute xyxy
        :param pro_features: (N, P, d_model) or None
        :param pooler: roi pooler
        :param time_emb: (N, d_model*4) time embedding
        :param use_cf: whether to use counterfactual attention instead of factual A
        :param cf_type: "random" | "uniform" | "shuffle" | "reversed"
        :return: class_logits [N,P,K], pred_bboxes [N,P,4], obj_features [1, N*P, C]
        """
        N, nr_boxes = bboxes.shape[:2]

        # roi feature
        proposal_boxes = [Boxes(bboxes[b]) for b in range(N)]

        # ---- DEBUG START: 检查 proposal_boxes 内容 ----
        for i, boxes in enumerate(proposal_boxes):
            try:
                t = boxes.tensor  # (num_boxes, 4)
            except Exception:
                # 如果 proposal_boxes 不是 Boxes 类型，打印信息
                print(f"DEBUG WARNING: proposal_boxes[{i}] is not Boxes, type:", type(boxes))
                continue
            if torch.isnan(t).any() or torch.isinf(t).any():
                print(f"DEBUG ERROR: proposal_boxes[{i}] has NaN/Inf")
                print("proposal_boxes sample (first 10 rows):", t[:10])
                raise RuntimeError("NaN/Inf in proposal_boxes")
            if (t.abs() > 1e8).any():
                print(f"DEBUG ERROR: proposal_boxes[{i}] extremely large values")
                print("proposal_boxes sample (first 10 rows):", t[:10])
                raise RuntimeError("proposal_boxes out-of-range")
        # ---- DEBUG END ----
        roi_features = pooler(features, proposal_boxes)  # (out_h*out_w, N*P, C) transposed later

        if pro_features is None:
            pro_features = roi_features.view(N, nr_boxes, self.d_model, -1).mean(-1)  # GAP -> [N,P,C]

        roi_features = roi_features.view(N * nr_boxes, self.d_model, -1).permute(2, 0, 1)  # [S, N*P, C]

        # self-attention on proposals
        pro_features = pro_features.view(N, nr_boxes, self.d_model).permute(1, 0, 2)      # [P, N, C]
        pro_features2 = self.self_attn(pro_features, pro_features, value=pro_features)[0]
        pro_features = pro_features + self.dropout1(pro_features2)
        pro_features = self.norm1(pro_features)

        # instance-level dynamic conv interaction
        pro_features = pro_features.view(nr_boxes, N, self.d_model).permute(1, 0, 2).reshape(1, N * nr_boxes, self.d_model)
        pro_features2 = self.inst_interact(pro_features, roi_features)                     # [1, N*P, C]
        pro_features = pro_features + self.dropout2(pro_features2)
        obj_features = self.norm2(pro_features)

        # FFN
        obj_features2 = self.linear2(self.dropout(self.activation(self.linear1(obj_features))))
        obj_features = obj_features + self.dropout3(obj_features2)
        obj_features = self.norm3(obj_features)

        # === CAL gating on proposal features ===
        pfeat = obj_features.transpose(0, 1).reshape(N, nr_boxes, self.d_model)            # [N,P,C]
        if self.cal_on:
            A = self.cal_gate(pfeat)  # factual attention [N,P,C] in (0,1)
            if use_cf:
                # build counterfactual attention (no grad by default)
                if cf_type == "random":
                    Abar = torch.rand_like(A)
                elif cf_type == "uniform":
                    Abar = A.mean(dim=-1, keepdim=True).expand_as(A)
                elif cf_type == "shuffle":
                    B, P, C = A.shape
                    flat = A.reshape(B * P, C)
                    idx = torch.randperm(B * P, device=A.device)
                    Abar = flat[idx].reshape(B, P, C)
                elif cf_type == "reversed":
                    Abar = 1.0 - A
                else:
                    Abar = torch.rand_like(A)
                if Abar.requires_grad:
                    Abar = Abar.detach()
                pfeat_used = (pfeat.detach() if self.cal_detach_cf else pfeat) * Abar
            else:
                pfeat_used = pfeat * A
        else:
            pfeat_used = pfeat

        # linear heads
        fc_feature = pfeat_used.reshape(N * nr_boxes, -1)

        # time modulation
        scale_shift = self.block_time_mlp(time_emb)
        scale_shift = torch.repeat_interleave(scale_shift, nr_boxes, dim=0)
        scale, shift = scale_shift.chunk(2, dim=1)
        fc_feature = fc_feature * (scale + 1) + shift

        cls_feature = fc_feature.clone()
        reg_feature = fc_feature.clone()
        for cls_layer in self.cls_module:
            cls_feature = cls_layer(cls_feature)
        for reg_layer in self.reg_module:
            reg_feature = reg_layer(reg_feature)

        class_logits = self.class_logits(cls_feature)
        bboxes_deltas = self.bboxes_delta(reg_feature)
        pred_bboxes = self.apply_deltas(bboxes_deltas, bboxes.view(-1, 4))
        
        def _sanitize_xyxy(boxes):
            boxes = torch.nan_to_num(boxes, nan=0.0, posinf=1e4, neginf=-1e4)
            # 逐列修正 (x1,y1,x2,y2)
            x1 = torch.min(boxes[:, 0::4], boxes[:, 2::4])
            y1 = torch.min(boxes[:, 1::4], boxes[:, 3::4])
            x2 = torch.max(boxes[:, 0::4], boxes[:, 2::4])
            y2 = torch.max(boxes[:, 1::4], boxes[:, 3::4])
            out = torch.zeros_like(boxes)
            out[:, 0::4] = x1
            out[:, 1::4] = y1
            out[:, 2::4] = x2
            out[:, 3::4] = y2
            return out
        
        pred_bboxes = _sanitize_xyxy(pred_bboxes)
    
        return class_logits.view(N, nr_boxes, -1), pred_bboxes.view(N, nr_boxes, -1), obj_features

    def apply_deltas(self, deltas, boxes):
        """
        Apply transformation `deltas` (dx, dy, dw, dh) to `boxes`.
        Args:
            deltas: (N, 4)
            boxes : (N, 4) absolute xyxy
        """
        boxes = boxes.to(deltas.dtype)

        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        ctr_x = boxes[:, 0] + 0.5 * widths
        ctr_y = boxes[:, 1] + 0.5 * heights

        wx, wy, ww, wh = self.bbox_weights
        dx = deltas[:, 0::4] / wx
        dy = deltas[:, 1::4] / wy
        dw = deltas[:, 2::4] / ww
        dh = deltas[:, 3::4] / wh

        dw = torch.clamp(dw, max=self.scale_clamp)
        dh = torch.clamp(dh, max=self.scale_clamp)

        pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
        pred_ctr_y = dy * heights[:, None] + ctr_y[:, None]
        pred_w = torch.exp(dw) * widths[:, None]
        pred_h = torch.exp(dh) * heights[:, None]

        pred_boxes = torch.zeros_like(deltas)
        pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w  # x1
        pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h  # y1
        pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w  # x2
        pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h  # y2

        return pred_boxes


class DynamicConv(nn.Module):
    """
    Dynamic convolution to fuse proposal token with pooled RoI features.
    """
    def __init__(self, cfg):
        super().__init__()

        self.hidden_dim = cfg.MODEL.DiffusionDet.HIDDEN_DIM
        self.dim_dynamic = cfg.MODEL.DiffusionDet.DIM_DYNAMIC
        self.num_dynamic = cfg.MODEL.DiffusionDet.NUM_DYNAMIC
        self.num_params = self.hidden_dim * self.dim_dynamic
        self.dynamic_layer = nn.Linear(self.hidden_dim, self.num_dynamic * self.num_params)

        self.norm1 = nn.LayerNorm(self.dim_dynamic)
        self.norm2 = nn.LayerNorm(self.hidden_dim)

        self.activation = nn.ReLU(inplace=True)

        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        num_output = self.hidden_dim * pooler_resolution ** 2
        self.out_layer = nn.Linear(num_output, self.hidden_dim)
        self.norm3 = nn.LayerNorm(self.hidden_dim)

    def forward(self, pro_features, roi_features):
        '''
        pro_features: (1,  N * nr_boxes, self.hidden_dim)
        roi_features: (S,  N * nr_boxes, self.hidden_dim)
        '''
        features = roi_features.permute(1, 0, 2)
        parameters = self.dynamic_layer(pro_features).permute(1, 0, 2)

        param1 = parameters[:, :, :self.num_params].view(-1, self.hidden_dim, self.dim_dynamic)
        param2 = parameters[:, :, self.num_params:].view(-1, self.dim_dynamic, self.hidden_dim)

        features = torch.bmm(features, param1)
        features = self.norm1(features)
        features = self.activation(features)

        features = torch.bmm(features, param2)
        features = self.norm2(features)
        features = self.activation(features)

        features = features.flatten(1)
        features = self.out_layer(features)
        features = self.norm3(features)
        features = self.activation(features)

        return features


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")