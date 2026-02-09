# ========================================
# Modified by Shoufa Chen
# ========================================
# Modified by Peize Sun, Rufeng Zhang
# Contact: {sunpeize, cxrfzhang}@foxmail.com
#
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import copy
import logging
import numpy as np
import torch
import os
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T


__all__ = ["DiffusionDetDatasetMapper"]


def build_transform_gen(cfg, is_train):
    """
    Create a list of :class:`TransformGen` from config.
    Returns:
        list[TransformGen]
    """
    if is_train:
        min_size = cfg.INPUT.MIN_SIZE_TRAIN
        max_size = cfg.INPUT.MAX_SIZE_TRAIN
        sample_style = cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING
    else:
        min_size = cfg.INPUT.MIN_SIZE_TEST
        max_size = cfg.INPUT.MAX_SIZE_TEST
        sample_style = "choice"
    if sample_style == "range":
        assert len(min_size) == 2, "more than 2 ({}) min_size(s) are provided for ranges".format(len(min_size))

    logger = logging.getLogger(__name__)
    tfm_gens = []
    if is_train:
        tfm_gens.append(T.RandomFlip())
    # ResizeShortestEdge
    tfm_gens.append(T.ResizeShortestEdge(min_size, max_size, sample_style))

    if is_train:
        logger.info("TransformGens used in training: " + str(tfm_gens))
    return tfm_gens


class DiffusionDetDatasetMapper:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by DiffusionDet.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies geometric transforms to the image and annotation
    3. Find and applies suitable cropping to the image and annotation
    4. Prepare image and annotation to Tensors
    """

    def __init__(self, cfg, is_train=True):
        if cfg.INPUT.CROP.ENABLED and is_train:
            self.crop_gen = [
                T.ResizeShortestEdge([400, 500, 600], sample_style="choice"),
                T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE),
            ]
        else:
            self.crop_gen = None

        self.tfm_gens = build_transform_gen(cfg, is_train)
        logging.getLogger(__name__).info(
            "Full TransformGens used in training: {}, crop: {}".format(str(self.tfm_gens), str(self.crop_gen))
        )

        self.img_format = cfg.INPUT.FORMAT
        self.is_train = is_train

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, image)

        if self.crop_gen is None:
            image, transforms = T.apply_transform_gens(self.tfm_gens, image)
        else:
            if np.random.rand() > 0.5:
                image, transforms = T.apply_transform_gens(self.tfm_gens, image)
            else:
                image, transforms = T.apply_transform_gens(
                    self.tfm_gens[:-1] + self.crop_gen + self.tfm_gens[-1:], image
                )

        image_shape = image.shape[:2]  # h, w

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))

        if not self.is_train:
            # USER: Modify this if you want to keep them for some reason.
            dataset_dict.pop("annotations", None)
            return dataset_dict

        if "annotations" in dataset_dict:
            # USER: Modify this if you want to keep them for some reason.
            for anno in dataset_dict["annotations"]:
                anno.pop("segmentation", None)
                anno.pop("keypoints", None)

            # USER: Implement additional transformations if you have other types of data
            annos = [
                utils.transform_instance_annotations(obj, transforms, image_shape)
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            instances = utils.annotations_to_instances(annos, image_shape)
            dataset_dict["instances"] = utils.filter_empty_instances(instances)
        # 处理增强图像
        if self.is_train and "aug_file_names" in dataset_dict:
            aug_fns = dataset_dict.get("aug_file_names", [])
            if aug_fns:
                aug_images = []
                aug_gt_classes = []
                for fn in aug_fns:
                    # 支持相对路径：相对于原图路径解析
                    if not os.path.isabs(fn):
                        base_dir = os.path.dirname(dataset_dict.get("file_name", ""))
                        fn = os.path.join(base_dir, fn) if base_dir else fn

                    if not os.path.exists(fn):
                        logging.getLogger(__name__).warning(f"aug file not found: {fn}")
                        continue

                    try:
                        aug_img = utils.read_image(fn, format=self.img_format)
                    except Exception as e:
                        logging.getLogger(__name__).warning(f"failed to read aug image {fn}: {e}")
                        continue

                    # 重用与主图相同的变换流水线
                    if self.crop_gen is None:
                        aug_img, _ = T.apply_transform_gens(self.tfm_gens, aug_img)
                    else:
                        if np.random.rand() > 0.5:
                            aug_img, _ = T.apply_transform_gens(self.tfm_gens, aug_img)
                        else:
                            aug_img, _ = T.apply_transform_gens(
                                self.tfm_gens[:-1] + self.crop_gen + self.tfm_gens[-1:], aug_img
                            )

                    aug_images.append(torch.as_tensor(np.ascontiguousarray(aug_img.transpose(2, 0, 1))))

                    # 如果存在实例信息，复用原图的 gt_classes（与用户要求一致）
                    if "instances" in dataset_dict and getattr(dataset_dict["instances"], "gt_classes",
                                                               None) is not None:
                        try:
                            aug_gt_classes.append(dataset_dict["instances"].gt_classes.clone())
                        except Exception:
                            aug_gt_classes.append(torch.zeros((0,), dtype=torch.long))
                    else:
                        aug_gt_classes.append(torch.zeros((0,), dtype=torch.long))

                dataset_dict["aug_images"] = aug_images
                dataset_dict["aug_gt_classes"] = aug_gt_classes
        return dataset_dict