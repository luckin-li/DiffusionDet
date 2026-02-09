# ==========================================
# Modified by Shoufa Chen
# ===========================================
# Modified by Peize Sun, Rufeng Zhang
# Contact: {sunpeize, cxrfzhang}@foxmail.com
#
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DiffusionDet Training Script.

This script is a simplified version of the training script in detectron2/tools.
"""

import os
import itertools
import weakref
from typing import Any, Dict, List, Set
import logging
from collections import OrderedDict

import torch
from fvcore.nn.precise_bn import get_bn_modules

import detectron2.utils.comm as comm
from detectron2.utils.logger import setup_logger
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import build_detection_test_loader
from detectron2.data import build_detection_train_loader
from detectron2.engine import (
    DefaultTrainer,
    default_argument_parser,
    default_setup,
    launch,
    create_ddp_model,
    AMPTrainer,
    SimpleTrainer,
    hooks,
)
from detectron2.evaluation import COCOEvaluator, LVISEvaluator, verify_results
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.modeling import build_model

from diffusiondet import (
    DiffusionDetDatasetMapper,
    add_diffusiondet_config,
    DiffusionDetWithTTA,
)
from diffusiondet.util.model_ema import (
    add_model_ema_configs,
    may_build_model_ema,
    may_get_ema_checkpointer,
    EMAHook,
    apply_model_ema_and_restore,
    EMADetectionCheckpointer,
)

# ===== extra imports for dataset/visualization =====
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.coco import load_coco_json
from detectron2.utils.visualizer import Visualizer
import cv2  # used by checkout_dataset_annotation

# ===================================================

# CLASS_NAMES = ['001.Black_footed_Albatross', '002.Laysan_Albatross', '003.Sooty_Albatross', '004.Groove_billed_Ani', '005.Crested_Auklet', '006.Least_Auklet', '007.Parakeet_Auklet', '008.Rhinoceros_Auklet', '009.Brewer_Blackbird', '010.Red_winged_Blackbird', '011.Rusty_Blackbird', '012.Yellow_headed_Blackbird', '013.Bobolink', '014.Indigo_Bunting', '015.Lazuli_Bunting', '016.Painted_Bunting', '017.Cardinal', '018.Spotted_Catbird', '019.Gray_Catbird', '020.Yellow_breasted_Chat', '021.Eastern_Towhee', '022.Chuck_will_Widow', '023.Brandt_Cormorant', '024.Red_faced_Cormorant', '025.Pelagic_Cormorant', '026.Bronzed_Cowbird', '027.Shiny_Cowbird', '028.Brown_Creeper', '029.American_Crow', '030.Fish_Crow', '031.Black_billed_Cuckoo', '032.Mangrove_Cuckoo', '033.Yellow_billed_Cuckoo', '034.Gray_crowned_Rosy_Finch', '035.Purple_Finch', '036.Northern_Flicker', '037.Acadian_Flycatcher', '038.Great_Crested_Flycatcher', '039.Least_Flycatcher', '040.Olive_sided_Flycatcher', '041.Scissor_tailed_Flycatcher', '042.Vermilion_Flycatcher', '043.Yellow_bellied_Flycatcher', '044.Frigatebird', '045.Northern_Fulmar', '046.Gadwall', '047.American_Goldfinch', '048.European_Goldfinch', '049.Boat_tailed_Grackle', '050.Eared_Grebe', '051.Horned_Grebe', '052.Pied_billed_Grebe', '053.Western_Grebe', '054.Blue_Grosbeak', '055.Evening_Grosbeak', '056.Pine_Grosbeak', '057.Rose_breasted_Grosbeak', '058.Pigeon_Guillemot', '059.California_Gull', '060.Glaucous_winged_Gull', '061.Heermann_Gull', '062.Herring_Gull', '063.Ivory_Gull', '064.Ring_billed_Gull', '065.Slaty_backed_Gull', '066.Western_Gull', '067.Anna_Hummingbird', '068.Ruby_throated_Hummingbird', '069.Rufous_Hummingbird', '070.Green_Violetear', '071.Long_tailed_Jaeger', '072.Pomarine_Jaeger', '073.Blue_Jay', '074.Florida_Jay', '075.Green_Jay', '076.Dark_eyed_Junco', '077.Tropical_Kingbird', '078.Gray_Kingbird', '079.Belted_Kingfisher', '080.Green_Kingfisher', '081.Pied_Kingfisher', '082.Ringed_Kingfisher', '083.White_breasted_Kingfisher', '084.Red_legged_Kittiwake', '085.Horned_Lark', '086.Pacific_Loon', '087.Mallard', '088.Western_Meadowlark', '089.Hooded_Merganser', '090.Red_breasted_Merganser', '091.Mockingbird', '092.Nighthawk', '093.Clark_Nutcracker', '094.White_breasted_Nuthatch', '095.Baltimore_Oriole', '096.Hooded_Oriole', '097.Orchard_Oriole', '098.Scott_Oriole', '099.Ovenbird', '100.Brown_Pelican', '101.White_Pelican', '102.Western_Wood_Pewee', '103.Sayornis', '104.American_Pipit', '105.Whip_poor_Will', '106.Horned_Puffin', '107.Common_Raven', '108.White_necked_Raven', '109.American_Redstart', '110.Geococcyx', '111.Loggerhead_Shrike', '112.Great_Grey_Shrike', '113.Baird_Sparrow', '114.Black_throated_Sparrow', '115.Brewer_Sparrow', '116.Chipping_Sparrow', '117.Clay_colored_Sparrow', '118.House_Sparrow', '119.Field_Sparrow', '120.Fox_Sparrow', '121.Grasshopper_Sparrow', '122.Harris_Sparrow', '123.Henslow_Sparrow', '124.Le_Conte_Sparrow', '125.Lincoln_Sparrow', '126.Nelson_Sharp_tailed_Sparrow', '127.Savannah_Sparrow', '128.Seaside_Sparrow', '129.Song_Sparrow', '130.Tree_Sparrow', '131.Vesper_Sparrow', '132.White_crowned_Sparrow', '133.White_throated_Sparrow', '134.Cape_Glossy_Starling', '135.Bank_Swallow', '136.Barn_Swallow', '137.Cliff_Swallow', '138.Tree_Swallow', '139.Scarlet_Tanager', '140.Summer_Tanager', '141.Artic_Tern', '142.Black_Tern', '143.Caspian_Tern', '144.Common_Tern', '145.Elegant_Tern', '146.Forsters_Tern', '147.Least_Tern', '148.Green_tailed_Towhee', '149.Brown_Thrasher', '150.Sage_Thrasher', '151.Black_capped_Vireo', '152.Blue_headed_Vireo', '153.Philadelphia_Vireo', '154.Red_eyed_Vireo', '155.Warbling_Vireo', '156.White_eyed_Vireo', '157.Yellow_throated_Vireo', '158.Bay_breasted_Warbler', '159.Black_and_white_Warbler', '160.Black_throated_Blue_Warbler', '161.Blue_winged_Warbler', '162.Canada_Warbler', '163.Cape_May_Warbler', '164.Cerulean_Warbler', '165.Chestnut_sided_Warbler', '166.Golden_winged_Warbler', '167.Hooded_Warbler', '168.Kentucky_Warbler', '169.Magnolia_Warbler', '170.Mourning_Warbler', '171.Myrtle_Warbler', '172.Nashville_Warbler', '173.Orange_crowned_Warbler', '174.Palm_Warbler', '175.Pine_Warbler', '176.Prairie_Warbler', '177.Prothonotary_Warbler', '178.Swainson_Warbler', '179.Tennessee_Warbler', '180.Wilson_Warbler', '181.Worm_eating_Warbler', '182.Yellow_Warbler', '183.Northern_Waterthrush', '184.Louisiana_Waterthrush', '185.Bohemian_Waxwing', '186.Cedar_Waxwing', '187.American_Three_toed_Woodpecker', '188.Pileated_Woodpecker', '189.Red_bellied_Woodpecker', '190.Red_cockaded_Woodpecker', '191.Red_headed_Woodpecker', '192.Downy_Woodpecker', '193.Bewick_Wren', '194.Cactus_Wren', '195.Carolina_Wren', '196.House_Wren', '197.Marsh_Wren', '198.Rock_Wren', '199.Winter_Wren', '200.Common_Yellowthroat']

# CLASS_NAMES = [
#     "707-320",
#     "727-200",
#     "737-200",
#     "737-300",
#     "737-400",
#     "737-500",
#     "737-600",
#     "737-700",
#     "737-800",
#     "737-900",
#     "747-100",
#     "747-200",
#     "747-300",
#     "747-400",
#     "757-200",
#     "757-300",
#     "767-200",
#     "767-300",
#     "767-400",
#     "777-200",
#     "777-300",
#     "A300B4",
#     "A310",
#     "A318",
#     "A319",
#     "A320",
#     "A321",
#     "A330-200",
#     "A330-300",
#     "A340-200",
#     "A340-300",
#     "A340-500",
#     "A340-600",
#     "A380",
#     "ATR-42",
#     "ATR-72",
#     "An-12",
#     "BAE 146-200",
#     "BAE 146-300",
#     "BAE-125",
#     "Beechcraft 1900",
#     "Boeing 717",
#     "C-130",
#     "C-47",
#     "CRJ-200",
#     "CRJ-700",
#     "CRJ-900",
#     "Cessna 172",
#     "Cessna 208",
#     "Cessna 525",
#     "Cessna 560",
#     "Challenger 600",
#     "DC-10",
#     "DC-3",
#     "DC-6",
#     "DC-8",
#     "DC-9-30",
#     "DH-82",
#     "DHC-1",
#     "DHC-6",
#     "DHC-8-100",
#     "DHC-8-300",
#     "DR-400",
#     "Dornier 328",
#     "E-170",
#     "E-190",
#     "E-195",
#     "EMB-120",
#     "ERJ 135",
#     "ERJ 145",
#     "Embraer Legacy 600",
#     "Eurofighter Typhoon",
#     "F-16A/B",
#     "F/A-18",
#     "Falcon 2000",
#     "Falcon 900",
#     "Fokker 100",
#     "Fokker 50",
#     "Fokker 70",
#     "Global Express",
#     "Gulfstream IV",
#     "Gulfstream V",
#     "Hawk T1",
#     "Il-76",
#     "L-1011",
#     "MD-11",
#     "MD-80",
#     "MD-87",
#     "MD-90",
#     "Metroliner",
#     "Model B200",
#     "PA-28",
#     "SR-20",
#     "Saab 2000",
#     "Saab 340",
#     "Spitfire",
#     "Tornado",
#     "Tu-134",
#     "Tu-154",
#     "Yak-42",
# ]

CLASS_NAMES = [
    "bird",
    "octopus",
    "snake",
    "cat",
    "insect",
    "person",
    "fish",
    "lizard",
    "butterfly",
    "sea horse",
    "moth",
    "crocodile",
    "owl",
    "frog",
    "leopard",
    "shrimp",
    "deer",
    "wolf",
    "fox",
    "sea dragon",
    "squirrel",
    "spider",
    "sloth",
    "flatfish",
    "dog",
    "turtle",
    "bear",
    "rabbit",
    "weasel",
    "crab",
    "sheep",
    "dolphin",
    "tiger",
    "lion",
    "giraffe",
    "monkey",
    "seal",
    "camel",
    "kangaroo",
    "snail",
    "mouse",
    "hedgehog",
    "pig",
]

# 数据集路径
DATASET_ROOT = '/home/luckin/yangjiwei/CAMO-D'
# DATASET_ROOT = '/public_hw/home/cit_haochenliang/yangjiwei/cub_coco'
ANN_ROOT = os.path.join(DATASET_ROOT, 'annotations')

TRAIN_PATH = os.path.join(DATASET_ROOT, 'train')
VAL_PATH = os.path.join(DATASET_ROOT, 'test')
TEST_PATH = os.path.join(DATASET_ROOT, 'test')

TRAIN_JSON = os.path.join(ANN_ROOT, 'train.json')
VAL_JSON = os.path.join(ANN_ROOT, 'test.json')
TEST_JSON = os.path.join(ANN_ROOT, 'test.json')

# 声明数据集的子集
PREDEFINED_SPLITS_DATASET = {
    "train": (TRAIN_PATH, TRAIN_JSON),
    "val": (VAL_PATH, VAL_JSON),
    "test": (TEST_PATH, TEST_JSON),
}


# 注册数据集（方式一：基于预定义字典）
def register_dataset():
    """
    purpose: register all splits of dataset with PREDEFINED_SPLITS_DATASET
    """
    for key, (image_root, json_file) in PREDEFINED_SPLITS_DATASET.items():
        register_dataset_instances(name=key, json_file=json_file, image_root=image_root)


# 注册数据集实例
def register_dataset_instances(name, json_file, image_root):
    """
    purpose: register dataset to DatasetCatalog,
             register metadata to MetadataCatalog and set attribute
    """
    DatasetCatalog.register(name, lambda: load_coco_json(json_file, image_root, name))
    MetadataCatalog.get(name).set(
        json_file=json_file,
        image_root=image_root,
        evaluator_type="coco",
        thing_classes=CLASS_NAMES,
    )


# （方式二：plain 注册，保留以便需要时切换）
def plain_register_dataset():
    DatasetCatalog.register("coco_my_train", lambda: load_coco_json(TRAIN_JSON, TRAIN_PATH))
    MetadataCatalog.get("coco_my_train").set(
        thing_classes=CLASS_NAMES,
        evaluator_type='coco',
        json_file=TRAIN_JSON,
        image_root=TRAIN_PATH
    )
    DatasetCatalog.register("coco_my_val", lambda: load_coco_json(VAL_JSON, VAL_PATH))
    MetadataCatalog.get("coco_my_val").set(
        thing_classes=CLASS_NAMES,
        evaluator_type='coco',
        json_file=VAL_JSON,
        image_root=VAL_PATH
    )


# 可选：快速可视化检查标注
def checkout_dataset_annotation(name="coco_my_train", out_dir="out", max_vis=200):
    os.makedirs(out_dir, exist_ok=True)
    dataset_dicts = load_coco_json(TRAIN_JSON, TRAIN_PATH, name)
    print("num samples:", len(dataset_dicts))
    meta = MetadataCatalog.get(name)
    for i, d in enumerate(dataset_dicts):
        img = cv2.imread(d["file_name"])
        vis = Visualizer(img[:, :, ::-1], metadata=meta, scale=1.0).draw_dataset_dict(d)
        cv2.imwrite(os.path.join(out_dir, f"{i:06d}.jpg"), vis.get_image()[:, :, ::-1])
        if i + 1 >= max_vis:
            break


class Trainer(DefaultTrainer):
    """ Extension of the Trainer class adapted to DiffusionDet. """

    def __init__(self, cfg):
        """
        Args:
            cfg (CfgNode):
        """
        # call grandfather's __init__ while avoid father's __init__()
        nn = DefaultTrainer  # for clarity in comment above
        super(DefaultTrainer, self).__init__()

        logger = logging.getLogger("detectron2")
        if not logger.isEnabledFor(logging.INFO):  # setup_logger is not called for d2
            setup_logger()
        cfg = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size())

        # Assume these objects must be constructed in this order.
        model = self.build_model(cfg)
        optimizer = self.build_optimizer(cfg, model)
        data_loader = self.build_train_loader(cfg)

        model = create_ddp_model(model, broadcast_buffers=False)
        self._trainer = (AMPTrainer if cfg.SOLVER.AMP.ENABLED else SimpleTrainer)(
            model, data_loader, optimizer
        )

        self.scheduler = self.build_lr_scheduler(cfg, optimizer)

        ########## EMA ############
        kwargs = {'trainer': weakref.proxy(self)}
        kwargs.update(may_get_ema_checkpointer(cfg, model))
        self.checkpointer = DetectionCheckpointer(
            model,
            cfg.OUTPUT_DIR,
            **kwargs,
        )
        self.start_iter = 0
        self.max_iter = cfg.SOLVER.MAX_ITER
        self.cfg = cfg

        self.register_hooks(self.build_hooks())

    @classmethod
    def build_model(cls, cfg):
        model = build_model(cfg)
        logger = logging.getLogger(__name__)
        logger.info("Model:\n{}".format(model))
        # setup EMA
        may_build_model_ema(cfg, model)
        return model

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        if 'lvis' in dataset_name:
            return LVISEvaluator(dataset_name, cfg, True, output_folder)
        else:
            return COCOEvaluator(dataset_name, cfg, True, output_folder)

    @classmethod
    def build_train_loader(cls, cfg):
        mapper = DiffusionDetDatasetMapper(cfg, is_train=True)
        return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        return build_detection_test_loader(
            cfg, dataset_name, mapper=DiffusionDetDatasetMapper(cfg, is_train=False)
        )

    @classmethod
    def build_optimizer(cls, cfg, model):
        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()
        for key, value in model.named_parameters(recurse=True):
            if not value.requires_grad or value in memo:
                continue
            memo.add(value)
            lr = cfg.SOLVER.BASE_LR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY
            if "backbone" in key:
                lr = lr * cfg.SOLVER.BACKBONE_MULTIPLIER
            params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

        def maybe_add_full_model_gradient_clipping(optim):
            clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
            enable = (
                    cfg.SOLVER.CLIP_GRADIENTS.ENABLED
                    and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
                    and clip_norm_val > 0.0
            )

            class FullModelGradientClippingOptimizer(optim):
                def step(self, closure=None):
                    all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                    torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                    super().step(closure=closure)

            return FullModelGradientClippingOptimizer if enable else optimce

        optimizer_type = cfg.SOLVER.OPTIMIZER
        if optimizer_type == "SGD":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(
                params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM
            )
        elif optimizer_type == "ADAMW":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
                params, cfg.SOLVER.BASE_LR
            )
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")
        if not cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
            optimizer = maybe_add_gradient_clipping(cfg, optimizer)
        return optimizer

    @classmethod
    def ema_test(cls, cfg, model, evaluators=None):
        logger = logging.getLogger("detectron2.trainer")
        if cfg.MODEL_EMA.ENABLED:
            logger.info("Run evaluation with EMA.")
            with apply_model_ema_and_restore(model):
                results = cls.test(cfg, model, evaluators=evaluators)
        else:
            results = cls.test(cfg, model, evaluators=evaluators)
        return results

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        logger.info("Running inference with test-time augmentation ...")
        model = DiffusionDetWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        if cfg.MODEL_EMA.ENABLED:
            res = cls.ema_test(cfg, model, evaluators)  # FIX: ensure res is assigned
        else:
            res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res

    def build_hooks(self):
        cfg = self.cfg.clone()
        cfg.defrost()
        cfg.DATALOADER.NUM_WORKERS = 0  # save some memory and time for PreciseBN

        ret = [
            hooks.IterationTimer(),
            EMAHook(self.cfg, self.model) if cfg.MODEL_EMA.ENABLED else None,  # EMA hook
            hooks.LRScheduler(),
            hooks.PreciseBN(
                cfg.TEST.EVAL_PERIOD,
                self.model,
                self.build_train_loader(cfg),
                cfg.TEST.PRECISE_BN.NUM_ITER,
            )
            if cfg.TEST.PRECISE_BN.ENABLED and get_bn_modules(self.model)
            else None,
        ]

        if comm.is_main_process():
            ret.append(hooks.PeriodicCheckpointer(self.checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD))

        def test_and_save_results():
            self._last_eval_results = self.test(self.cfg, self.model)
            return self._last_eval_results

        ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD, test_and_save_results))

        if comm.is_main_process():
            ret.append(hooks.PeriodicWriter(self.build_writers(), period=20))
        return ret


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_diffusiondet_config(cfg)
    add_model_ema_configs(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)
    # 注册数据集（如需要 plain 版本，改为 plain_register_dataset()）
    register_dataset()

    if args.eval_only:
        model = Trainer.build_model(cfg)
        kwargs = may_get_ema_checkpointer(cfg, model)
        if cfg.MODEL_EMA.ENABLED:
            EMADetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR, **kwargs).resume_or_load(
                cfg.MODEL.WEIGHTS, resume=args.resume
            )
        else:
            DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR, **kwargs).resume_or_load(
                cfg.MODEL.WEIGHTS, resume=args.resume
            )
        res = Trainer.ema_test(cfg, model)
        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model))
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
