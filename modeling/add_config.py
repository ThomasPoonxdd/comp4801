# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from detectron2.config import CfgNode as CN


def add_vild_config(cfg):
    """
    Add config for densepose head.
    """
    _C = cfg

    _C.MODEL.CLIP_PATH = None

#     _C.MODEL.ROI_DENSEPOSE_HEAD = CN()
#     _C.MODEL.ROI_DENSEPOSE_HEAD.NAME = ""
#     _C.MODEL.ROI_DENSEPOSE_HEAD.NUM_STACKED_CONVS = 8
#     # Number of parts used for point labels
#     _C.MODEL.ROI_DENSEPOSE_HEAD.NUM_PATCHES = 24
#     _C.MODEL.ROI_DENSEPOSE_HEAD.DECONV_KERNEL = 4
#     _C.MODEL.ROI_DENSEPOSE_HEAD.CONV_HEAD_DIM = 512
#     _C.MODEL.ROI_DENSEPOSE_HEAD.CONV_HEAD_KERNEL = 3
#     _C.MODEL.ROI_DENSEPOSE_HEAD.UP_SCALE = 2
#     _C.MODEL.ROI_DENSEPOSE_HEAD.HEATMAP_SIZE = 56
#     _C.MODEL.ROI_DENSEPOSE_HEAD.POOLER_TYPE = "ROIAlignV2"
#     _C.MODEL.ROI_DENSEPOSE_HEAD.POOLER_RESOLUTION = 14
#     _C.MODEL.ROI_DENSEPOSE_HEAD.POOLER_SAMPLING_RATIO = 2
#     # Overlap threshold for an RoI to be considered foreground (if >= FG_IOU_THRESHOLD)
#     _C.MODEL.ROI_DENSEPOSE_HEAD.FG_IOU_THRESHOLD = 0.7
#     # Loss weights for annotation masks.(14 Parts)
#     _C.MODEL.ROI_DENSEPOSE_HEAD.INDEX_WEIGHTS = 2.0
#     # Loss weights for surface parts. (24 Parts)
#     _C.MODEL.ROI_DENSEPOSE_HEAD.PART_WEIGHTS = 0.3
#     # Loss weights for UV regression.
#     _C.MODEL.ROI_DENSEPOSE_HEAD.POINT_REGRESSION_WEIGHTS = 0.1