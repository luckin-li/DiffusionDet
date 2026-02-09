# ========================================
# Modified by Shoufa Chen
# ========================================
# Modified by Peize Sun, Rufeng Zhang
# Contact: {sunpeize, cxrfzhang}@foxmail.com
#
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from detectron2.config import CfgNode as CN


def add_diffusiondet_config(cfg):
    """
    Add config for DiffusionDet (+ SD residual-gated fusion)
    """
    cfg.MODEL.DiffusionDet = CN()
    cfg.MODEL.DiffusionDet.NUM_CLASSES = 80
    cfg.MODEL.DiffusionDet.NUM_PROPOSALS = 300
    cfg.MODEL.DiffusionDet.TEST_NUM_PROPOSALS = 2000

    # RCNN Head.
    cfg.MODEL.DiffusionDet.NHEADS = 8
    cfg.MODEL.DiffusionDet.DROPOUT = 0.0
    cfg.MODEL.DiffusionDet.DIM_FEEDFORWARD = 2048
    cfg.MODEL.DiffusionDet.ACTIVATION = 'relu'
    cfg.MODEL.DiffusionDet.HIDDEN_DIM = 256
    cfg.MODEL.DiffusionDet.NUM_CLS = 1
    cfg.MODEL.DiffusionDet.NUM_REG = 3
    cfg.MODEL.DiffusionDet.NUM_HEADS = 6

    # Dynamic Conv.
    cfg.MODEL.DiffusionDet.NUM_DYNAMIC = 2
    cfg.MODEL.DiffusionDet.DIM_DYNAMIC = 64

    # === CAL ===
    cfg.MODEL.DiffusionDet.CAL_ON = False
    cfg.MODEL.DiffusionDet.CAL_TYPES = ("random", "uniform", "shuffle")
    cfg.MODEL.DiffusionDet.CAL_MARGIN = 0.2      # hinge margin
    cfg.MODEL.DiffusionDet.CAL_LAMBDA = 1.0      # CAL loss 分类权重
    cfg.MODEL.DiffusionDet.CAL_DETACH_CF = True  # 反事实分支是否 detach
    cfg.MODEL.DiffusionDet.CAL_PROB = 1.0        # 每个 iteration 启用 CAL 的概率

    # === Loss ===
    cfg.MODEL.DiffusionDet.CLASS_WEIGHT = 1.0
    cfg.MODEL.DiffusionDet.GIOU_WEIGHT = 1.0
    cfg.MODEL.DiffusionDet.L1_WEIGHT = 0.05
    cfg.MODEL.DiffusionDet.DEEP_SUPERVISION = True
    cfg.MODEL.DiffusionDet.NO_OBJECT_WEIGHT = 0.1

    # === Focal Loss ===
    cfg.MODEL.DiffusionDet.USE_FOCAL = True
    cfg.MODEL.DiffusionDet.USE_FED_LOSS = False
    cfg.MODEL.DiffusionDet.ALPHA = 0.25
    cfg.MODEL.DiffusionDet.GAMMA = 2.0
    cfg.MODEL.DiffusionDet.PRIOR_PROB = 0.01

    # augmentation branch (新增)
    cfg.MODEL.DiffusionDet.AUG_ON = False
    cfg.MODEL.DiffusionDet.AUG_PROB = 1.0
    cfg.MODEL.DiffusionDet.AUG_LOSS_CE_WEIGHT = 1.0
    cfg.MODEL.DiffusionDet.AUG_NO_GRAD_BACKBONE = False

    # === Dynamic K ===
    cfg.MODEL.DiffusionDet.OTA_K = 5

    # === Diffusion ===
    cfg.MODEL.DiffusionDet.SNR_SCALE = 2.0
    cfg.MODEL.DiffusionDet.SAMPLE_STEP = 1

    # === Inference ===
    cfg.MODEL.DiffusionDet.USE_NMS = True

    # === Swin Backbones ===
    cfg.MODEL.SWIN = CN()
    cfg.MODEL.SWIN.SIZE = 'B'  # 'T', 'S', 'B'
    cfg.MODEL.SWIN.USE_CHECKPOINT = False
    cfg.MODEL.SWIN.OUT_FEATURES = (0, 1, 2, 3)  # modify

    # === Optimizer ===
    cfg.SOLVER.OPTIMIZER = "ADAMW"
    cfg.SOLVER.BACKBONE_MULTIPLIER = 1.0

    # === TTA ===
    cfg.TEST.AUG.MIN_SIZES = (400, 500, 600, 640, 700, 900, 1000, 1100, 1200, 1300, 1400, 1800, 800)
    cfg.TEST.AUG.CVPODS_TTA = True
    cfg.TEST.AUG.SCALE_FILTER = True
    cfg.TEST.AUG.SCALE_RANGES = ([96, 10000], [96, 10000],
                                 [64, 10000], [64, 10000],
                                 [64, 10000], [0, 10000],
                                 [0, 10000], [0, 256],
                                 [0, 256], [0, 192],
                                 [0, 192], [0, 96],
                                 [0, 10000])

    # =========================================================
    # SD Residual-Gated Fusion (v1) — 配置项
    # =========================================================
    cfg.MODEL.SDFUSE_ENABLE = False                # 是否启用 SD 残差门控融合
    cfg.MODEL.SDFUSE_CACHE_DIR = "/public_hw/home/cit_haochenliang/yangjiwei/dc_cache"  # 离线特征缓存目录（与 .pt 同名）
    cfg.MODEL.SDFUSE_DTYPE = "fp16"               # 融合分支使用的精度：fp16 / fp32

    # SD 特征通道；SD 2.1-base 常见为 320/640/1280（按 p8/p16/p32）
    cfg.MODEL.SDFUSE_P_CH = CN()
    cfg.MODEL.SDFUSE_P_CH.p8 = 320
    cfg.MODEL.SDFUSE_P_CH.p16 = 640
    cfg.MODEL.SDFUSE_P_CH.p32 = 1280
