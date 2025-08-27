#!/usr/bin/env python
# coding: utf-8

# In[1]:


import jittor as jt
from jittor import nn,Module,init
import json
import inspect
import collections
from functools import partial
from collections.abc import Sized
from collections import defaultdict
from jittor.dataset import Dataset
import jittor.transform as transform
import multiprocessing as mp
from PIL import Image
import os
import argparse
import os.path as osp
import numpy as np
from numpy import random
from copy import deepcopy
import copy
import math
from typing import List, Optional, Sequence, Tuple, Union, Type, TypeVar, Dict, Iterable, Literal, Any, Iterator, Generator, cast, Callable 
import cv2
import warnings
from contextlib import contextmanager
from io import BytesIO, StringIO
from pathlib import Path
import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Real
import pickle
import time
import itertools
import gc
from contextlib import contextmanager

T = TypeVar("T")

   


# In[ ]:





# In[2]:


# env settings

# Single-scale training is recommended to
# be turned on, which can speed up training.

jt.flags.use_cuda = 0 # todo wait to enable
cv2.setNumThreads(0)

#due to jittor use nccl by default and computational resources limited so there is no env_settings for disk_config
#and I donot find how to set this. Luckily remdet default env is nccl
# Todo: find how to switch disk_config

#Here are some basic settings(copied directly from the origin,fixed some parameters according to the ability of my computer)
# ========================Frequently modified parameters======================
# ----data related----
data_root = '/root/.cache/jittor/dataset/Fast Test/'# '/mnt/e/git clone/UAVDT/' # Root path of data
# Path of train annotation file
train_ann_file = 'ann/fast_test.json'# 'annotations/UAV-benchmark-M-Train.json'
train_data_prefix = 'images'# 'images/UAV-benchmark-M' # Prefix of train imagae path

# Path of val annotation file
val_ann_file = 'ann/fast_test.json' # annotations/UAV-benchmark-M-Val.json'
val_data_prefix = 'images' # 'images/UAV-benchmark-M' # Prefix of val image path

# here we use fast test to test if the module can run normally
# classes = ("pedestrian", "people", "bicycle", "car", "van", "truck", "tricycle", "awning-tricycle", "bus", "motor")

classes = (
    'car',
    'truck',
    'bus',
)

num_classes = len(classes)  # Number of classes for classification

# Batch size of a single GPU during training
train_batch_size_per_gpu = 2

# Worker to pre-fetch data for each single GPU during training
train_num_workers = 1

# persistent_workers must be False if num_workers is 0
persistent_workers = True

# todo need to konw this parameter usage

# -----train val related-----
# Base learning rate for optim_wrapper. Corresponding to 8xb16=64 bs
base_lr = 0.01
max_epochs = 10  # Maximum training epochs
# Disable mosaic augmentation for final 10 epochs (stage 2)
close_mosaic_epochs = 1

model_test_cfg = dict(
    # The config of multi-label for multi-class prediction.
    multi_label=True,
    # The number of boxes before NMS
    nms_pre=30000,
    score_thr=0.001,  # Threshold to filter out boxes.
    nms=dict(type='nms', iou_threshold=0.7),  # NMS type and threshold
    max_per_img=300)  # Max number of detections of each image

# ========================Possible modified parameters========================
# -----data related-----
img_scale = (640, 640)  # width, height

img_scales = [
    (
        640,
        640,
    ),
    (
        320,
        320,
    ),
    (
        960,
        960,
    ),
]

# Dataset type, this will be used to define the dataset
dataset_type = 'YOLOv5CocoDataset'
# Batch size of a single GPU during validation
val_batch_size_per_gpu = 2
# Worker to pre-fetch data for each single GPU during validation
val_num_workers = 1

batch_shapes_cfg = None

# -----model related-----
# The scaling factor that controls the depth of the network structure
deepen_factor = 0.33
# The scaling factor that controls the width of the network structure
widen_factor = 0.5
# Strides of multi-scale prior box
strides = [8, 16, 32]
# The output channel of the last stage
last_stage_out_channels = 1024
num_det_layers = 3  # The number of model output scales
norm_cfg = dict(type='BN', momentum=0.03, eps=0.001)  # Normalization config  for ddp
mixup_prob = 0.1

# -----train val related-----
affine_scale = 0.5  # YOLOv5RandomAffine scaling ratio
# YOLOv5RandomAffine aspect ratio of width and height thres to filter bboxes
max_aspect_ratio = 100
tal_topk = 10  # Number of bbox selected in each level
tal_alpha = 0.5  # A Hyper-parameter related to alignment_metrics
tal_beta = 6.0  # A Hyper-parameter related to alignment_metrics
# TODO: Automatically scale loss_weight based on number of detection layers
loss_cls_weight = 0.5
loss_bbox_weight = 7.5
# Since the dfloss is implemented differently in the official
# and mmdet, we're going to divide loss_weight by 4.
loss_dfl_weight = 1.5 / 4
lr_factor = 0.01  # Learning rate scaling factor
weight_decay = 0.0005
# Save model checkpoint and validation intervals in stage 1
save_epoch_intervals = 1
# validation intervals in stage 2
val_interval_stage2 = 1
# The maximum checkpoints to keep.
max_keep_ckpts = 2


# In[ ]:





# In[ ]:





# In[ ]:





# In[3]:


def is_abs(path:str) -> bool:
    if osp.isabs(path) or path.startswith(('http://', 'https://', 's3://')):
        return True
    else:
        return False
def _isArrayLike(obj):
    return hasattr(obj, '__iter__') and hasattr(obj, '__len__')
@contextmanager
def get_local_path(file_path :Union[str, Path]) -> Generator[Union[str, Path], None, None]:
    local_path = Path(file_path).absolute()
    yield local_path
def list_from_file(filename,
                   prefix='',
                   offset=0,
                   max_num=0,
                   encoding='utf-8',
                   file_client_args=None,
                   backend_args=None):
    cnt = 0
    item_list = []
    with open(filepath, encoding=encoding) as f:
        text = f.read()
    with StringIO(text) as f:
        for _ in range(offset):
            f.readline()
        for line in f:
            if 0 < max_num <= cnt:
                break
            item_list.append(prefix + line.rstrip('\n\r'))
            cnt += 1
    return item_list


# In[ ]:





# In[4]:


class COCO:
    def __init__(self, annotation_file=None):
        self.dataset,self.anns,self.cats,self.imgs = dict(),dict(),dict(),dict()
        self.imgToAnns, self.catToImgs = defaultdict(list), defaultdict(list)
        if not annotation_file == None:
            print('loading annotations into memory...')
            tic = time.time()
            dataset = json.load(open(annotation_file, 'r'))
            assert type(dataset)==dict, 'annotation file format {} not supported'.format(type(dataset))
            print('Done (t={:0.2f}s)'.format(time.time()- tic))
            self.dataset = dataset
            self.createIndex()
        self.img_ann_map = self.imgToAnns
        self.cat_img_map = self.catToImgs
    def createIndex(self):
        # create index
        print('creating index...')
        anns, cats, imgs = {}, {}, {}
        imgToAnns,catToImgs = defaultdict(list),defaultdict(list)
        if 'annotations' in self.dataset:
            for ann in self.dataset['annotations']:
                imgToAnns[ann['image_id']].append(ann)
                anns[ann['id']] = ann

        if 'images' in self.dataset:
            for img in self.dataset['images']:
                imgs[img['id']] = img

        if 'categories' in self.dataset:
            for cat in self.dataset['categories']:
                cats[cat['id']] = cat

        if 'annotations' in self.dataset and 'categories' in self.dataset:
            for ann in self.dataset['annotations']:
                catToImgs[ann['category_id']].append(ann['image_id'])

        print('index created!')

        # create class members
        self.anns = anns
        self.imgToAnns = imgToAnns
        self.catToImgs = catToImgs
        self.imgs = imgs
        self.cats = cats
    def info(self):
        """
        Print information about the annotation file.
        :return:
        """
        for key, value in self.dataset['info'].items():
            print('{}: {}'.format(key, value))
    def get_ann_ids(self, img_ids=[], cat_ids=[], area_rng=[], iscrowd=None):
        """
        Get ann ids that satisfy given filter conditions. default skips that filter
        :param img_ids  (int array)     : get anns for given imgs
               cat_ids  (int array)     : get anns for given cats
               area_rng (float array)   : get anns for given area range (e.g. [0 inf])
               iscrowd (boolean)       : get anns for given crowd label (False or True)
        :return: ids (int array)       : integer array of ann ids
        """
        img_ids = img_ids if _isArrayLike(img_ids) else [img_ids]
        cat_ids = cat_ids if _isArrayLike(cat_ids) else [cat_ids]

        if len(img_ids) == len(cat_ids) == len(area_rng) == 0:
            anns = self.dataset['annotations']
        else:
            if not len(img_ids) == 0:
                lists = [self.imgToAnns[imgId] for imgId in img_ids if imgId in self.imgToAnns]
                anns = list(itertools.chain.from_iterable(lists))
            else:
                anns = self.dataset['annotations']
            anns = anns if len(cat_ids)  == 0 else [ann for ann in anns if ann['category_id'] in cat_ids]
            anns = anns if len(area_rng) == 0 else [ann for ann in anns if ann['area'] > area_rng[0] and ann['area'] < area_rng[1]]
        if not iscrowd == None:
            ids = [ann['id'] for ann in anns if ann['iscrowd'] == iscrowd]
        else:
            ids = [ann['id'] for ann in anns]
        return ids
    def get_cat_ids(self, cat_names=[], sup_names=[], cat_ids=[]):
        """
        filtering parameters. default skips that filter.
        :param cat_name (str array)  : get cats for given cat names
        :param sup_name (str array)  : get cats for given supercategory names
        :param cat_ids (int array)  : get cats for given cat ids
        :return: ids (int array)   : integer array of cat ids
        """
        cat_name = cat_names if _isArrayLike(cat_names) else [cat_names]
        sup_name = sup_names if _isArrayLike(sup_names) else [sup_names]
        cat_ids = cat_ids if _isArrayLike(cat_ids) else [cat_ids]

        if len(cat_name) == len(sup_name) == len(cat_ids) == 0:
            cats = self.dataset['categories']
        else:
            cats = self.dataset['categories']
            cats = cats if len(cat_name) == 0 else [cat for cat in cats if cat['name']          in cat_name]
            cats = cats if len(sup_name) == 0 else [cat for cat in cats if cat['supercategory'] in sup_name]
            cats = cats if len(cat_ids) == 0 else [cat for cat in cats if cat['id']            in cat_ids]
        ids = [cat['id'] for cat in cats]
        return ids

    def get_img_ids(self, img_ids=[], cat_ids=[]):
        '''
        Get img ids that satisfy given filter conditions.
        :param img_ids (int array) : get imgs for given ids
        :param cat_ids (int array) : get imgs with all given cats
        :return: ids (int array)  : integer array of img ids
        '''
        img_ids = img_ids if _isArrayLike(img_ids) else [img_ids]
        cat_ids = cat_ids if _isArrayLike(cat_ids) else [cat_ids]

        if len(img_ids) == len(cat_ids) == 0:
            ids = self.imgs.keys()
        else:
            ids = set(img_ids)
            for i, catId in enumerate(cat_ids):
                if i == 0 and len(ids) == 0:
                    ids = set(self.catToImgs[catId])
                else:
                    ids &= set(self.catToImgs[catId])
        return list(ids)

    def load_anns(self, ids=[]):
        """
        Load anns with the specified ids.
        :param ids (int array)       : integer ids specifying anns
        :return: anns (object array) : loaded ann objects
        """
        if _isArrayLike(ids):
            return [self.anns[id] for id in ids]
        elif type(ids) == int:
            return [self.anns[ids]]

    def load_cats(self, ids=[]):
        """
        Load cats with the specified ids.
        :param ids (int array)       : integer ids specifying cats
        :return: cats (object array) : loaded cat objects
        """
        if _isArrayLike(ids):
            return [self.cats[id] for id in ids]
        elif type(ids) == int:
            return [self.cats[ids]]

    def load_imgs(self, ids=[]):
        """
        Load anns with the specified ids.
        :param ids (int array)       : integer ids specifying img
        :return: imgs (object array) : loaded img objects
        """
        if _isArrayLike(ids):
            return [self.imgs[id] for id in ids]
        elif type(ids) == int:
            return [self.imgs[ids]]


# In[ ]:





# In[ ]:





# In[5]:


# ===============================Unmodified in most cases====================
# The frame used
# Note:most of them copied from origin but fixed a little to adatapt jittor frame
# dataloder
# class Dataloder(Dataset):
#     def __init__(self, data_root, ann_file, data_prefix,batch_size,num_workers,classes,transform = None):
#         super().__init__
# #         self.data_root = data_root
#         self.img_file = data_root + data_prefix
#         self.ann_file = data_root + ann_file
# #         self.data_prefix = data_prefix
#         self.batch_size = batch_size
#         self.transform = transform
#         self.num_workers = num_workers
#         self.classes = classes
#         self.cat_classes_id = {v:k for v,k in enumerate(self.classes)}
#         img_exts = set(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'))
#         self.imgs = []
        
#         with open(self.ann_file, 'r', encoding = 'utf-8') as f:
#             self.data = json.load(f)
            
        
#     def load_img(self):
# #         img_file = self.data_root + self.data_prefix
# #         for root, _dir, imgs in sorted(os.walk(img_file)):
# #             for img in sorted(imgs):
# #                 if os.path.splittext(img)[-1].lower() in img_exts:
# #                     path = os.path.join(img_file,img)
# #                     sef.imgs[img] = path
# #         for i,img in self.data['images']:
# #             self.img['id'] = img['id']
#         self.imgs = seld.data['images']
#         self.set_attrs(total_len=len(self.imgs))
        
# #     def load_ann(self):
# #         ann_file = self.data_root + self.ann_file
# #         with open(ann_file, 'r', encoding = 'utf-8') as f:
# #             ann = json.load(f)
# #     def 
#     def __getitem__(self,k):
# #         data_prefix = self.data_root + self.
#         with open(self.img_file + '/' + self.imgs[k]['file_name'], 'rb') as f:
#             image = Image.open(f).convert('RGB')
#             if self.transform:
#                 image = self.transform(image)
# #             return image, self.ann[]
#         pass


class LoadYOLOAnnotations:
    """Because the YOLO series does not need to consider ignore bboxes for the
    time being, in order to speed up the pipeline, it can be excluded in
    advance.

    Args:
        mask2bbox (bool): Whether to use mask annotation to get bbox.
            Defaults to False.
        poly2mask (bool): Whether to transform the polygons to bitmaps.
            Defaults to False.
        merge_polygons (bool): Whether to merge polygons into one polygon.
            If merged, the storage structure is simpler and training is more
            effcient, especially if the mask inside a bbox is divided into
            multiple polygons. Defaults to True.
    """

    def __init__(self,
                 mask2bbox: bool = False,
                 poly2mask: bool = False,
                 merge_polygons: bool = True,
                 with_mask: bool = False,
                 box_type: str = 'hbox',
                 # use for semseg
                 reduce_zero_label: bool = False,
                 ignore_index: int = 255,
                 with_bbox: bool = True,
                 with_label: bool = True,
                 with_seg: bool = False,
                 with_keypoints: bool = False,
                 imdecode_backend: str = 'cv2',
                 file_client_args: Optional[dict] = None,
                 *,
                 backend_args: Optional[dict] = None
                     ):
        self.mask2bbox = mask2bbox
        self.merge_polygons = merge_polygons
        self.with_bbox = with_bbox
        self.with_label = with_label
        self.with_seg = with_seg
        self.with_keypoints = with_keypoints
        self.imdecode_backend = imdecode_backend
        self.with_mask = with_mask
        self.poly2mask = poly2mask
        self.box_type = box_type
        self.reduce_zero_label = reduce_zero_label
        self.ignore_index = ignore_index
        self.file_client_args: Optional[dict] = None
        self.backend_args: Optional[dict] = None
        assert not poly2mask, 'Does not support BitmapMasks considering '                               'that bitmap consumes more memory.'
        if self.mask2bbox:
            assert self.with_mask, 'Using mask2bbox requires '                                    'with_mask is True.'
        self._mask_ignore_flag = None
        
    def __call__(self, results: dict) -> dict:
        """Function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:``mmengine.BaseDataset``.

        Returns:
            dict: The dict contains loaded bounding box, label and
            semantic segmentation.
        """
        if self.mask2bbox:
            self._load_masks(results)
            if self.with_label:
                self._load_labels(results)
                self._update_mask_ignore_data(results)
            gt_bboxes = results['gt_masks'].get_bboxes(dst_type='hbox')
            results['gt_bboxes'] = gt_bboxes
        elif self.with_keypoints:
            self._load_kps(results)
            _, box_type_cls = get_box_type(self.box_type)
            results['gt_bboxes'] = box_type_cls(
                results.get('bbox', []), dtype=torch.float32)
        else:
            if self.with_bbox:
                self._load_bboxes(results)
            if self.with_label:
                self._load_labels(results)
            if self.with_seg:
                self._load_seg_map(results)
            if self.with_keypoints:
                self._load_kps(results)
            self._update_mask_ignore_data(results)
        return results

    def _update_mask_ignore_data(self, results: dict) -> None:
        if 'gt_masks' not in results:
            return

        if 'gt_bboxes_labels' in results and len(
                results['gt_bboxes_labels']) != len(results['gt_masks']):
            assert len(results['gt_bboxes_labels']) == len(
                self._mask_ignore_flag)
            results['gt_bboxes_labels'] = results['gt_bboxes_labels'][
                self._mask_ignore_flag]

        if 'gt_bboxes' in results and len(results['gt_bboxes']) != len(
                results['gt_masks']):
            assert len(results['gt_bboxes']) == len(self._mask_ignore_flag)
            results['gt_bboxes'] = results['gt_bboxes'][self._mask_ignore_flag]

    def _load_bboxes(self, results: dict):
        """Private function to load bounding box annotations.
        Note: BBoxes with ignore_flag of 1 is not considered.
        Args:
            results (dict): Result dict from :obj:``mmengine.BaseDataset``.

        Returns:
            dict: The dict contains loaded bounding box annotations.
        """
        gt_bboxes = []
        gt_ignore_flags = []
        for instance in results.get('instances', []):
            if instance['ignore_flag'] == 0:
                gt_bboxes.append(instance['bbox'])
                gt_ignore_flags.append(instance['ignore_flag'])
        results['gt_ignore_flags'] = np.array(gt_ignore_flags, dtype=bool)

        if self.box_type is None:
            results['gt_bboxes'] = np.array(
                gt_bboxes, dtype=np.float32).reshape((-1, 4))
        else:
            raise NotImplementedError('not support bbox type')
#             _, box_type_cls = get_box_type(self.box_type)
#             results['gt_bboxes'] = box_type_cls(gt_bboxes, dtype=torch.float32)

    def _load_labels(self, results: dict):
        """Private function to load label annotations.

        Note: BBoxes with ignore_flag of 1 is not considered.
        Args:
            results (dict): Result dict from :obj:``mmengine.BaseDataset``.
        Returns:
            dict: The dict contains loaded label annotations.
        """
        gt_bboxes_labels = []
        for instance in results.get('instances', []):
            if instance['ignore_flag'] == 0:
                gt_bboxes_labels.append(instance['bbox_label'])
        results['gt_bboxes_labels'] = np.array(
            gt_bboxes_labels, dtype=np.int64)

    def _load_masks(self, results: dict) -> None:
        """Private function to load mask annotations.

        Args:
            results (dict): Result dict from :obj:``mmengine.BaseDataset``.
        """
        gt_masks = []
        gt_ignore_flags = []
        self._mask_ignore_flag = []
        for instance in results.get('instances', []):
            if instance['ignore_flag'] == 0:
                if 'mask' in instance:
                    gt_mask = instance['mask']
                    if isinstance(gt_mask, list):
                        gt_mask = [
                            np.array(polygon) for polygon in gt_mask
                            if len(polygon) % 2 == 0 and len(polygon) >= 6
                        ]
                        if len(gt_mask) == 0:
                            # ignore
                            self._mask_ignore_flag.append(0)
                        else:
                            if len(gt_mask) > 1 and self.merge_polygons:
                                gt_mask = self.merge_multi_segment(gt_mask)
                            gt_masks.append(gt_mask)
                            gt_ignore_flags.append(instance['ignore_flag'])
                            self._mask_ignore_flag.append(1)
                    else:
                        raise NotImplementedError(
                            'Only supports mask annotations in polygon '
                            'format currently')
                else:
                    # TODO: Actually, gt with bbox and without mask needs
                    #  to be retained
                    self._mask_ignore_flag.append(0)
        self._mask_ignore_flag = np.array(self._mask_ignore_flag, dtype=bool)
        results['gt_ignore_flags'] = np.array(gt_ignore_flags, dtype=bool)

        h, w = results['ori_shape']
        raise NotImplementedError(
            'only used bbox ,not support mask'
        )
#         gt_masks = PolygonMasks([mask for mask in gt_masks], h, w)
        results['gt_masks'] = gt_masks

    def merge_multi_segment(self,
                            gt_masks: List[np.ndarray]) -> List[np.ndarray]:
        """Merge multi segments to one list.

        Find the coordinates with min distance between each segment,
        then connect these coordinates with one thin line to merge all
        segments into one.
        Args:
            gt_masks(List(np.array)):
                original segmentations in coco's json file.
                like [segmentation1, segmentation2,...],
                each segmentation is a list of coordinates.
        Return:
            gt_masks(List(np.array)): merged gt_masks
        """
        s = []
        segments = [np.array(i).reshape(-1, 2) for i in gt_masks]
        idx_list = [[] for _ in range(len(gt_masks))]

        # record the indexes with min distance between each segment
        for i in range(1, len(segments)):
            idx1, idx2 = self.min_index(segments[i - 1], segments[i])
            idx_list[i - 1].append(idx1)
            idx_list[i].append(idx2)

        # use two round to connect all the segments
        # first round: first to end, i.e. A->B(partial)->C
        # second round: end to first, i.e. C->B(remaining)-A
        for k in range(2):
            # forward first round
            if k == 0:
                for i, idx in enumerate(idx_list):
                    # middle segments have two indexes
                    # reverse the index of middle segments
                    if len(idx) == 2 and idx[0] > idx[1]:
                        idx = idx[::-1]
                        segments[i] = segments[i][::-1, :]
                    # add the idx[0] point for connect next segment
                    segments[i] = np.roll(segments[i], -idx[0], axis=0)
                    segments[i] = np.concatenate(
                        [segments[i], segments[i][:1]])
                    # deal with the first segment and the last one
                    if i in [0, len(idx_list) - 1]:
                        s.append(segments[i])
                    # deal with the middle segment
                    # Note that in the first round, only partial segment
                    # are appended.
                    else:
                        idx = [0, idx[1] - idx[0]]
                        s.append(segments[i][idx[0]:idx[1] + 1])
            # forward second round
            else:
                for i in range(len(idx_list) - 1, -1, -1):
                    # deal with the middle segment
                    # append the remaining points
                    if i not in [0, len(idx_list) - 1]:
                        idx = idx_list[i]
                        nidx = abs(idx[1] - idx[0])
                        s.append(segments[i][nidx:])
        return [np.concatenate(s).reshape(-1, )]

    def min_index(self, arr1: np.ndarray, arr2: np.ndarray) -> Tuple[int, int]:
        """Find a pair of indexes with the shortest distance.

        Args:
            arr1: (N, 2).
            arr2: (M, 2).
        Return:
            tuple: a pair of indexes.
        """
        dis = ((arr1[:, None, :] - arr2[None, :, :]) ** 2).sum(-1)
        return np.unravel_index(np.argmin(dis, axis=None), dis.shape)

    def _load_kps(self, results: dict) -> None:
        """Private function to load keypoints annotations.

        Args:
            results (dict): Result dict from
                :class:`mmengine.dataset.BaseDataset`.

        Returns:
            dict: The dict contains loaded keypoints annotations.
        """
        results['height'] = results['img_shape'][0]
        results['width'] = results['img_shape'][1]
        num_instances = len(results.get('bbox', []))

        if num_instances == 0:
            results['keypoints'] = np.empty(
                (0, len(results['flip_indices']), 2), dtype=np.float32)
            results['keypoints_visible'] = np.empty(
                (0, len(results['flip_indices'])), dtype=np.int32)
            results['category_id'] = []

        results['gt_keypoints'] = Keypoints(
            keypoints=results['keypoints'],
            keypoints_visible=results['keypoints_visible'],
            flip_indices=results['flip_indices'],
        )

        results['gt_ignore_flags'] = np.array([False] * num_instances)
        results['gt_bboxes_labels'] = np.array(results['category_id']) - 1

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(with_bbox={self.with_bbox}, '
        repr_str += f'with_label={self.with_label}, '
        repr_str += f'with_mask={self.with_mask}, '
        repr_str += f'with_seg={self.with_seg}, '
        repr_str += f'mask2bbox={self.mask2bbox}, '
        repr_str += f'poly2mask={self.poly2mask}, '
        repr_str += f"imdecode_backend='{self.imdecode_backend}', "
        repr_str += f'backend_args={self.backend_args})'
        return repr_str


# In[ ]:





# In[6]:


#Notcie here I decide to make the data in resault to be np.ndarray(some function did not support jt.Var)
def bbox_clip(bboxes,area):
    width ,height = enumerate(area)
    bboxes[...,0] = bboxes[...,0].clip(min=0)
    bboxes[...,1] = bboxes[...,1].clip(min=0)
    bboxes[...,2] = bboxes[...,2].clip(max=width)
    bboxes[...,3] = bboxes[...,3].clip(max=height)
    return bboxes
def bbox_rescale(bboxes:np.ndarray,scale_factor):
    if scale_factor is not None:
        sx, sy = scale_factor[0], scale_factor[1]
        scale = np.array([sx, sy, sx, sy])
        bboxes = bboxes / scale
    return bboxes
def _fixed_scale_size(
        size: Tuple[int, int],
        scale: Union[float, int, tuple],
) -> Tuple[int, int]:
    """Rescale a size by a ratio.

    Args:
        size (tuple[int]): (w, h).
        scale (float | tuple(float)): Scaling factor.

    Returns:
        tuple[int]: scaled size.
    """
    if isinstance(scale, (float, int)):
        scale = (scale, scale)
    w, h = size
    # don't need o.5 offset
    return int(w * float(scale[0])), int(h * float(scale[1]))


def rescale_size(old_size: tuple,
                 scale: Union[float, int, tuple],
                 return_scale: bool = False) -> tuple:
    """Calculate the new size to be rescaled to.

    Args:
        old_size (tuple[int]): The old size (w, h) of image.
        scale (float | tuple[int]): The scaling factor or maximum size.
            If it is a float number, then the image will be rescaled by this
            factor, else if it is a tuple of 2 integers, then the image will
            be rescaled as large as possible within the scale.
        return_scale (bool): Whether to return the scaling factor besides the
            rescaled image size.

    Returns:
        tuple[int]: The new rescaled image size.
    """
    w, h = old_size
    if isinstance(scale, (float, int)):
        if scale <= 0:
            raise ValueError(f'Invalid scale {scale}, must be positive.')
        scale_factor = scale
    elif isinstance(scale, tuple):
        max_long_edge = max(scale)
        max_short_edge = min(scale)
        scale_factor = min(max_long_edge / max(h, w),
                           max_short_edge / min(h, w))
    else:
        raise TypeError(
            f'Scale must be a number or tuple of int, but got {type(scale)}')
    # only change this
    new_size = _fixed_scale_size((w, h), scale_factor)

    if return_scale:
        return new_size, scale_factor
    else:
        return new_size


def imrescale(
        img: np.ndarray,
        scale: Union[float, Tuple[int, int]],
        return_scale: bool = False,
        interpolation: str = 'bilinear',
        backend: Optional[str] = None
) -> Union[np.ndarray, Tuple[np.ndarray, float]]:
    """Resize image while keeping the aspect ratio.

    Args:
        img (ndarray): The input image.
        scale (float | tuple[int]): The scaling factor or maximum size.
            If it is a float number, then the image will be rescaled by this
            factor, else if it is a tuple of 2 integers, then the image will
            be rescaled as large as possible within the scale.
        return_scale (bool): Whether to return the scaling factor besides the
            rescaled image.
        interpolation (str): Same as :func:`resize`.
        backend (str | None): Same as :func:`resize`.

    Returns:
        ndarray: The rescaled image.
    """
    h, w = img.shape[:2]
    new_size, scale_factor = rescale_size((w, h), scale, return_scale=True)
#     rescaled_img = imresize(
#         img, new_size, interpolation=interpolation, backend=backend)
    rescaled_img = transform.resize(
        img = Image.fromarray(img), size = (new_size[1], new_size[0]), interpolation=interpolation
    )
    rescaled_img = rescaled_img.numpy()
    if return_scale:
        return rescaled_img, scale_factor
    else:
        return rescaled_img
    
def bbox_translate(
    bboxes:np.ndarray, distance:Tuple[float, float]
):
    assert len(distance) == 2
    bboxes = bboxes + np.array(distance).reshape(1, 1, 2)
    return bboxes
# NOTE:in oringin source code img should be a np.ndarray # TODO
def mask_translate(mask:np.ndarray, out_shape, offest, direction,border_value: Union[int, tuple] = 0):
    '''
    out shape should be (w,h)
    '''
    directions = ['horizontal', 'vertical']
    assert direction in directions
    oringin_dim = mask.ndim
    if oringin_dim == 2:
        mask = np.expand_dims(mask, -1)
    H, W, C = mask.shape
    mask = transform.resize(img = mask, size = (out_shape[1], out_shape[0]), interpolation='bilinear')
    W, H = out_shape
    if direction == 'horizontal':
#         translate_matrix = np.float32([[1, 0, offset], [0, 1, 0]])
        translate_matrix = (1, 0, offset, 0, 1, 0)
    elif direction == 'vertical':
#         translate_matrix = np.float32([[1, 0, 0], [0, 1, offset]])
        translate_matrix = (1, 0, 0, 0, 1, offset)
    img = Image.fromarray(mask)
    img = img.transform(
        size=(W,H),
        method=Image.AFFINE,
        data=translate_matrix,
        fillcolor=border_value[:3]
    )
    return np.array(img)
def impad(img: np.ndarray,
          *,
          shape: Optional[Tuple[int, int]] = None,
          padding: Union[int, tuple, None] = None,
          pad_val: Union[float, List] = 0,
          padding_mode: str = 'constant') -> np.ndarray:
    """Pad the given image to a certain shape or pad on all sides with
    specified padding mode and padding value.

    Args:
        img (ndarray): Image to be padded.
        shape (tuple[int]): Expected padding shape (w, h). Default: None.
        padding (int or tuple[int]): Padding on each border. If a single int is
            provided this is used to pad all borders. If tuple of length 2 is
            provided this is the padding on left/right and top/bottom
            respectively. If a tuple of length 4 is provided this is the
            padding for the left, top, right and bottom borders respectively.
            Default: None. Note that `shape` and `padding` can not be both
            set.
        pad_val (Number | Sequence[Number]): Values to be filled in padding
            areas when padding_mode is 'constant'. Default: 0.
        padding_mode (str): Type of padding. Should be: constant, edge,
            reflect or symmetric. Default: constant.

            - constant: pads with a constant value, this value is specified
              with pad_val.
            - edge: pads with the last value at the edge of the image.
            - reflect: pads with reflection of image without repeating the last
              value on the edge. For example, padding [1, 2, 3, 4] with 2
              elements on both sides in reflect mode will result in
              [3, 2, 1, 2, 3, 4, 3, 2].
            - symmetric: pads with reflection of image repeating the last value
              on the edge. For example, padding [1, 2, 3, 4] with 2 elements on
              both sides in symmetric mode will result in
              [2, 1, 1, 2, 3, 4, 4, 3]

    Returns:
        ndarray: The padded image.
    """

    assert (shape is not None) ^ (padding is not None)
    if shape is not None:
        width = max(shape[0] - img.shape[0], 0)
        height = max(shape[1] - img.shape[1], 0)
        padding = (0, 0, width, height)

    # check pad_val
    if isinstance(pad_val, tuple):
        assert len(pad_val) == img.shape[-1]
    elif not isinstance(pad_val, numbers.Number):
        raise TypeError('pad_val must be a int or a tuple. '
                        f'But received {type(pad_val)}')

    # check padding
    if isinstance(padding, tuple) and len(padding) in [2, 4]:
        if len(padding) == 2:
            padding = (padding[0], padding[1], padding[0], padding[1])
    elif isinstance(padding, numbers.Number):
        padding = (padding, padding, padding, padding)
    else:
        raise ValueError('Padding must be a int or a 2, or 4 element tuple.'
                         f'But received {padding}')

    # check padding mode
    assert padding_mode in ['constant', 'edge', 'reflect', 'symmetric']

    border_type = {
        'constant',
        'edge',
        'reflect',
        'symmetric'
    }
    pad = ((padding[1],        padding[3]),         (padding[0],         padding[2]),(0, 0))
    if padding_mode == 'constant':
        img = np.pad(
            img = img,
            pad_width = pad,
            mode = padding_mode
        )
    else:
        img = np.pad(
            img,
            pad_width = pad,
            mode = mode
        )
#     img = cv2.copyMakeBorder(
#         img,
#         padding[1],
#         padding[3],
#         padding[0],
#         padding[2],
#         border_type[padding_mode],
#         value=pad_val)

    return img
def imflip(img: np.ndarray, direction: str = 'horizontal') -> np.ndarray:
    """Flip an image horizontally or vertically.

    Args:
        img (ndarray): Image to be flipped.
        direction (str): The flip direction, either "horizontal" or
            "vertical" or "diagonal".

    Returns:
        ndarray: The flipped image.
    """
    assert direction in ['horizontal', 'vertical', 'diagonal']
    if direction == 'horizontal':
        return np.flip(img, axis=1)
    elif direction == 'vertical':
        return np.flip(img, axis=0)
    else:
        return np.flip(img, axis=(0, 1))
def sample_odd_from_range(random_state, low: int, high: int) -> int:
    """Sample an odd number from the range [low, high] (inclusive).

    Args:
        random_state (random.Random): instance of random.Random
        low (int): lower bound (will be converted to nearest valid odd number)
        high (int): upper bound (will be converted to nearest valid odd number)

    Returns:
        int: Randomly sampled odd number from the range

    Note:
        - Input values will be converted to nearest valid odd numbers:
          * Values less than 3 will become 3
          * Even values will be rounded up to next odd number
        - After normalization, high must be >= low

    """
    # Normalize low value
    low = max(3, low + (low % 2 == 0))
    # Normalize high value
    high = max(3, high + (high % 2 == 0))

    # Ensure high >= low after normalization
    high = max(high, low)

    if low == high:
        return low

    # Calculate number of possible odd values
    num_odd_values = (high - low) // 2 + 1
    # Generate random index and convert to corresponding odd number
    rand_idx = random_state.randint(0, num_odd_values - 1)
    return low + (2 * rand_idx)
def blur(img: np.ndarray, ksize: int) -> np.ndarray:
    array = jt.array(img)
    in_channel = img.shape[2]
    out_channel = in_channel
    padding = (ksize // 2, ksize // 2)
    kernel = jt.ones((out_channel,in_channel,ksize,ksize)) / (ksize * ksize)
    conv = nn.Conv(
        in_channel = in_channel,
        out_channel = out_channel,
        kernel_size = kernel,
        padding = padding,
        bias = False
    )
    conv.weight = jt.nn.Parameter(kernel, need_grad = False)
    blurred_img = conv(array).numpy()
    return blurred_img
def median_blur(img: np.ndarray, ksize: int) -> np.ndarray:
    assert ksize % 2 == 1
    padding = ksize // 2
#     array = jt.array(img)
    h,w = img.shape[:2]
    c = img.shape[2] if len(img.shape) == 3 else 1
    padding = ksize // 2
    if channels == 3:
        img = np.pad(img,(padding,padding),(padding,padding),(0,0),mode = 'edge')
    else:
        img = np.pad(img,(padding,padding),(padding,padding),mode = 'edge')
    array = jt.array(img)
    array = array.unfold(2,ksize,1).unfold(3,ksize,1)
    array.reshape(array.shape[:4] + (-1,))
    array.float()
    array = jt.sort(array, dim = -1)
    array = array[..., (ksize * ksize) // 2]
    return array.numpy().astype(np.unit8)
def f(t):
    delta = 6 / 29
    return np.where( t > delta ** 3,
                    t ** (1/3),
                    (t / (3 * delta ** 2)) + (4 / 29)
                   )
def gamma_correction(t):
    return np.where(
        t > 0.0031308,
        1.055 * (t ** (1/2.4)) - 0.055,
        t * 12.92
    )
def rgb2lab(img:np.ndarray):
    img = img.astype(np.float32) / 255.0
    img = np.where(
        img > 0.04045,
        ((img + 0.055) / 1.055) ** 2.4,
        rgb / 12.92
    )
    matrix = np.array(
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041]
    )
    xyz = np.tensordot(img, matrix.T, axes = 1)
    xn, yn, zn = 0.95047, 1.0, 1.08883
    x = xyz[...,0] / xn
    y = xyz[...,0] / yn
    z = xyz[...,0] / zn
    fx = f(x)
    fy = f(y)
    fz = f(z)
    L = 116 * fy -16
    a = 500 * (fx - fy)
    b = 200 * (fy - fz)
    lab =  np.stack([L , a, b],axis = -1)
    return lab
def lab2rgb(img:np.ndarray):
    L = img[..., 0]
    a = img[..., 1]
    b = img[..., 2]
    H, W = L.shape()
    xn, yn ,zn = 95.047, 100.0, 108.883
    L = L / 100.0
    fL = f(L)
    x = xn * f(fL + a /500.0) / xn
    y = yn * fL / yn
    z = zn * f(fL - b / 200.0) / zn
    matrix = np.array([
        [3.2406, -1.5372, -0.4986],
        [-0.9689, 1.8758, 0.0415],
        [0.0557, -0.2040, 1.0570]
    ])
    xyz = np.stack([x, y, z], axis=-1)
    rgb = np.dot(xyz, matrix.T)
    rgb = gamma_correction(rgb)
    rgb = np.clip(rgb, 0.0, 1.0)
    rgb = (rgb * 255).astype(np.uint8)
    return rgb
def to_gray(img:np.ndarray,num_output_channels: int,mode):
    assert num_output_channels > 0
    assert mode in ["weighted_average",
            "from_lab",
            "desaturation",
            "average",
            "max",
            "pca",]
    if mode == 'weighted_average':
        image = Image.fromarray(img).convert('L')
        img = np.array(image)
    elif mode == 'max':
        raise NotImplementedError
    elif mode == 'average':
        raise NotImplementedError
    elif mode == 'from_lab':
        raise NotImplementedError
    elif mode == 'pca':
        raise NotImplementedError
    # If output should be single channel, add channel dimension if needed
    if num_output_channels == 1:
        return img

    squeezed = np.squeeze(img)
    # For multi-channel output, use tile for better performance
    return np.tile(squeezed[..., np.newaxis], (1,) * squeezed.ndim + (num_output_channels,))
def clahe(
    img: np.ndarray,
    clip_limit: float,
    tile_grid_size: Tuple[int, int],
) -> np.ndarray:
    clahe_mat = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    if is_grayscale_image(img):
        return clahe_mat.apply(img)

    img_lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    img_lab[:, :, 0] = clahe_mat.apply(img_lab[:, :, 0])

    return cv2.cvtColor(img_lab, cv2.COLOR_LAB2RGB)


# In[ ]:





# In[7]:


def _pillow2array(img,
                  flag: str = 'color',
                  channel_order: str = 'bgr') -> np.ndarray:
    """Convert a pillow image to numpy array.

    Args:
        img (:obj:`PIL.Image.Image`): The image loaded using PIL
        flag (str): Flags specifying the color type of a loaded image,
            candidates are 'color', 'grayscale' and 'unchanged'.
            Default to 'color'.
        channel_order (str): The channel order of the output image array,
            candidates are 'bgr' and 'rgb'. Default to 'bgr'.

    Returns:
        np.ndarray: The converted numpy array
    """
    channel_order = channel_order.lower()
    if channel_order not in ['rgb', 'bgr']:
        raise ValueError('channel order must be either "rgb" or "bgr"')

    if flag == 'unchanged':
        array = np.array(img)
        if array.ndim >= 3 and array.shape[2] >= 3:  # color image
            array[:, :, :3] = array[:, :, (2, 1, 0)]  # RGB to BGR
    else:
        # Handle exif orientation tag
        if flag in ['color', 'grayscale']:
            img = ImageOps.exif_transpose(img)
        # If the image mode is not 'RGB', convert it to 'RGB' first.
        if img.mode != 'RGB':
            if img.mode != 'LA':
                # Most formats except 'LA' can be directly converted to RGB
                img = img.convert('RGB')
            else:
                # When the mode is 'LA', the default conversion will fill in
                #  the canvas with black, which sometimes shadows black objects
                #  in the foreground.
                #
                # Therefore, a random color (124, 117, 104) is used for canvas
                img_rgba = img.convert('RGBA')
                img = Image.new('RGB', img_rgba.size, (124, 117, 104))
                img.paste(img_rgba, mask=img_rgba.split()[3])  # 3 is alpha
        if flag in ['color', 'color_ignore_orientation']:
            array = np.array(img)
            if channel_order != 'rgb':
                array = array[:, :, ::-1]  # RGB to BGR
        elif flag in ['grayscale', 'grayscale_ignore_orientation']:
            img = img.convert('L')
            array = np.array(img)
        else:
            raise ValueError(
                'flag must be "color", "grayscale", "unchanged", '
                f'"color_ignore_orientation" or "grayscale_ignore_orientation"'
                f' but got {flag}')
    return array


# In[ ]:





# In[8]:


def imfrombytes(content: bytes,
                flag: str = 'color',
                channel_order: str = 'bgr',
                backend: Optional[str] = None) -> np.ndarray:
    """Read an image from bytes.

    Args:
        content (bytes): Image bytes got from files or other streams.
        flag (str): Same as :func:`imread`.
        channel_order (str): The channel order of the output, candidates
            are 'bgr' and 'rgb'. Default to 'bgr'.
        backend (str | None): The image decoding backend type. Options are
            `cv2`, `pillow`, `turbojpeg`, `tifffile`, `None`. If backend is
            None, the global imread_backend specified by ``mmcv.use_backend()``
            will be used. Default: None.

    Returns:
        ndarray: Loaded image array.

    Examples:
        >>> img_path = '/path/to/img.jpg'
        >>> with open(img_path, 'rb') as f:
        >>>     img_buff = f.read()
        >>> img = mmcv.imfrombytes(img_buff)
        >>> img = mmcv.imfrombytes(img_buff, flag='color', channel_order='rgb')
        >>> img = mmcv.imfrombytes(img_buff, backend='pillow')
        >>> img = mmcv.imfrombytes(img_buff, backend='cv2')
    """

    if backend is None:
        backend = imread_backend
    if backend not in supported_backends:
        raise ValueError(
            f'backend: {backend} is not supported. Supported '
            "backends are 'cv2', 'turbojpeg', 'pillow', 'tifffile'")
    if backend == 'turbojpeg':
#         img = jpeg.decode(  # type: ignore
#             content, _jpegflag(flag, channel_order))
#         if img.shape[-1] == 1:
#             img = img[:, :, 0]
#         return img
        raise NotImplementedError('please import related lib,and uncomment this part')
    elif backend == 'pillow':
        with io.BytesIO(content) as buff:
            img = Image.open(buff)
            img = _pillow2array(img, flag, channel_order)
        return img
    elif backend == 'tifffile':
#         with io.BytesIO(content) as buff:
#             img = tifffile.imread(buff)
#         return img
        raise NotImplementedError('please import related lib,and uncomment this part')
    else:
        img_np = np.frombuffer(content, np.uint8)
        flag = imread_flags[flag] if is_str(flag) else flag
        img = cv2.imdecode(img_np, flag)
        if flag == IMREAD_COLOR and channel_order == 'rgb':
            cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)
        return img


# In[9]:


def to_tensor(img:np.ndarray):
    return transform.ToTensor(img)
def get_shape(data: Dict[str, Any]):    
    if "image" in data:
        if isinstance(data["image"], np.ndarray):
            height, width = data["image"].shape[:2]
            return {"height": height, "width": width}
        else:
            raise RuntimeError(f"Unsupported image type: {type(img)}")
    if "images" in data:
        if isinstance(data["image"][0], np.ndarray):
            height, width = data["image"][0].shape[:2]
            return {"height": height, "width": width}
        else:
            raise RuntimeError(f"Unsupported image type: {type(img)}")
    raise ValueError("No image or volume found in data", data.keys())


# In[10]:


class LoadImageFromFile:
    """Load an image from file.

    Required Keys:

    - img_path

    Modified Keys:

    - img
    - img_shape
    - ori_shape

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        color_type (str): The flag argument for :func:`mmcv.imfrombytes`.
            Defaults to 'color'.
        imdecode_backend (str): The image decoding backend type. The backend
            argument for :func:`mmcv.imfrombytes`.
            See :func:`mmcv.imfrombytes` for details.
            Defaults to 'cv2'.
        file_client_args (dict, optional): Arguments to instantiate a
            FileClient. See :class:`mmengine.fileio.FileClient` for details.
            Defaults to None. It will be deprecated in future. Please use
            ``backend_args`` instead.
            Deprecated in version 2.0.0rc4.
        ignore_empty (bool): Whether to allow loading empty image or file path
            not existent. Defaults to False.
        backend_args (dict, optional): Instantiates the corresponding file
            backend. It may contain `backend` key to specify the file
            backend. If it contains, the file backend corresponding to this
            value will be used and initialized with the remaining values,
            otherwise the corresponding file backend will be selected
            based on the prefix of the file path. Defaults to None.
            New in version 2.0.0rc4.
    """

    def __init__(self,
                 to_float32: bool = False,
                 color_type: str = 'color',
                 imdecode_backend: str = 'cv2',
                 file_client_args: Optional[dict] = None,
                 ignore_empty: bool = False,
                 *,
                 backend_args: Optional[dict] = None) -> None:
        self.ignore_empty = ignore_empty
        self.to_float32 = to_float32
        self.color_type = color_type
        self.imdecode_backend = imdecode_backend

        self.file_client_args: Optional[dict] = None
        self.backend_args: Optional[dict] = None
        if file_client_args is not None:
            warnings.warn(
                '"file_client_args" will be deprecated in future. '
                'Please use "backend_args" instead', DeprecationWarning)
            if backend_args is not None:
                raise ValueError(
                    '"file_client_args" and "backend_args" cannot be set '
                    'at the same time.')

            self.file_client_args = file_client_args.copy()
        if backend_args is not None:
            self.backend_args = backend_args.copy()

    def __call__(self, results: dict) -> Optional[dict]:
        """Functions to load image.

        Args:
            results (dict): Result dict from
                :class:`mmengine.dataset.BaseDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """

        filename = results['img_path']
        try:
            if self.file_client_args is not None:
#                 file_client = fileio.FileClient.infer_client(
#                     self.file_client_args, filename)
#                 img_bytes = file_client.get(filename)
                raise NotImplementedError
            else:
                assert backend_args is None, 'only support remdet framework which means backend_args should be none'
#                 img_bytes = fileio.get(
#                     filename, backend_args=self.backend_args)
                # notice: only support local backend
                with open(filename,'rb') as f:
                    img_bytes = f.read()
            img = imfrombytes(
                img_bytes, flag=self.color_type, backend=self.imdecode_backend)
        except Exception as e:
            if self.ignore_empty:
                return None
            else:
                raise e
        # in some cases, images are not read successfully, the img would be
        # `None`, refer to https://github.com/open-mmlab/mmpretrain/issues/1427
        assert img is not None, f'failed to load image: {filename}'
        if self.to_float32:
            img = img.astype(np.float32)

        results['img'] = img
        results['img_shape'] = img.shape[:2]
        results['ori_shape'] = img.shape[:2]
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'ignore_empty={self.ignore_empty}, '
                    f'to_float32={self.to_float32}, '
                    f"color_type='{self.color_type}', "
                    f"imdecode_backend='{self.imdecode_backend}', ")

        if self.file_client_args is not None:
            repr_str += f'file_client_args={self.file_client_args})'
        else:
            repr_str += f'backend_args={self.backend_args})'

        return repr_str


# In[ ]:





# In[11]:


class BaseDataElement:
    """A base data interface that supports Tensor-like and dict-like
    operations.

    A typical data elements refer to predicted results or ground truth labels
    on a task, such as predicted bboxes, instance masks, semantic
    segmentation masks, etc. Because groundtruth labels and predicted results
    often have similar properties (for example, the predicted bboxes and the
    groundtruth bboxes), MMEngine uses the same abstract data interface to
    encapsulate predicted results and groundtruth labels, and it is recommended
    to use different name conventions to distinguish them, such as using
    ``gt_instances`` and ``pred_instances`` to distinguish between labels and
    predicted results. Additionally, we distinguish data elements at instance
    level, pixel level, and label level. Each of these types has its own
    characteristics. Therefore, MMEngine defines the base class
    ``BaseDataElement``, and implement ``InstanceData``, ``PixelData``, and
    ``LabelData`` inheriting from ``BaseDataElement`` to represent different
    types of ground truth labels or predictions.

    Another common data element is sample data. A sample data consists of input
    data (such as an image) and its annotations and predictions. In general,
    an image can have multiple types of annotations and/or predictions at the
    same time (for example, both pixel-level semantic segmentation annotations
    and instance-level detection bboxes annotations). All labels and
    predictions of a training sample are often passed between Dataset, Model,
    Visualizer, and Evaluator components. In order to simplify the interface
    between components, we can treat them as a large data element and
    encapsulate them. Such data elements are generally called XXDataSample in
    the OpenMMLab. Therefore, Similar to `nn.Module`, the `BaseDataElement`
    allows `BaseDataElement` as its attribute. Such a class generally
    encapsulates all the data of a sample in the algorithm library, and its
    attributes generally are various types of data elements. For example,
    MMDetection is assigned by the BaseDataElement to encapsulate all the data
    elements of the sample labeling and prediction of a sample in the
    algorithm library.

    The attributes in ``BaseDataElement`` are divided into two parts,
    the ``metainfo`` and the ``data`` respectively.

        - ``metainfo``: Usually contains the
          information about the image such as filename,
          image_shape, pad_shape, etc. The attributes can be accessed or
          modified by dict-like or object-like operations, such as
          ``.`` (for data access and modification), ``in``, ``del``,
          ``pop(str)``, ``get(str)``, ``metainfo_keys()``,
          ``metainfo_values()``, ``metainfo_items()``, ``set_metainfo()`` (for
          set or change key-value pairs in metainfo).

        - ``data``: Annotations or model predictions are
          stored. The attributes can be accessed or modified by
          dict-like or object-like operations, such as
          ``.``, ``in``, ``del``, ``pop(str)``, ``get(str)``, ``keys()``,
          ``values()``, ``items()``. Users can also apply tensor-like
          methods to all :obj:`jt.Var` in the ``data_fields``,
          such as ``.cuda()``, ``.cpu()``, ``.numpy()``, ``.to()``,
          ``to_tensor()``, ``.detach()``.

    Args:
        metainfo (dict, optional): A dict contains the meta information
            of single image, such as ``dict(img_shape=(512, 512, 3),
            scale_factor=(1, 1, 1, 1))``. Defaults to None.
        kwargs (dict, optional): A dict contains annotations of single image or
            model predictions. Defaults to None.

    Examples:
        >>> import torch
        >>> from mmengine.structures import BaseDataElement
        >>> gt_instances = BaseDataElement()
        >>> bboxes = jt.rand((5, 4))
        >>> scores = jt.rand((5,))
        >>> img_id = 0
        >>> img_shape = (800, 1333)
        >>> gt_instances = BaseDataElement(
        ...     metainfo=dict(img_id=img_id, img_shape=img_shape),
        ...     bboxes=bboxes, scores=scores)
        >>> gt_instances = BaseDataElement(
        ...     metainfo=dict(img_id=img_id, img_shape=(640, 640)))

        >>> # new
        >>> gt_instances1 = gt_instances.new(
        ...     metainfo=dict(img_id=1, img_shape=(640, 640)),
        ...                   bboxes=jt.rand((5, 4)),
        ...                   scores=jt.rand((5,)))
        >>> gt_instances2 = gt_instances1.new()

        >>> # add and process property
        >>> gt_instances = BaseDataElement()
        >>> gt_instances.set_metainfo(dict(img_id=9, img_shape=(100, 100)))
        >>> assert 'img_shape' in gt_instances.metainfo_keys()
        >>> assert 'img_shape' in gt_instances
        >>> assert 'img_shape' not in gt_instances.keys()
        >>> assert 'img_shape' in gt_instances.all_keys()
        >>> print(gt_instances.img_shape)
        (100, 100)
        >>> gt_instances.scores = jt.rand((5,))
        >>> assert 'scores' in gt_instances.keys()
        >>> assert 'scores' in gt_instances
        >>> assert 'scores' in gt_instances.all_keys()
        >>> assert 'scores' not in gt_instances.metainfo_keys()
        >>> print(gt_instances.scores)
        tensor([0.5230, 0.7885, 0.2426, 0.3911, 0.4876])
        >>> gt_instances.bboxes = jt.rand((5, 4))
        >>> assert 'bboxes' in gt_instances.keys()
        >>> assert 'bboxes' in gt_instances
        >>> assert 'bboxes' in gt_instances.all_keys()
        >>> assert 'bboxes' not in gt_instances.metainfo_keys()
        >>> print(gt_instances.bboxes)
        tensor([[0.0900, 0.0424, 0.1755, 0.4469],
                [0.8648, 0.0592, 0.3484, 0.0913],
                [0.5808, 0.1909, 0.6165, 0.7088],
                [0.5490, 0.4209, 0.9416, 0.2374],
                [0.3652, 0.1218, 0.8805, 0.7523]])

        >>> # delete and change property
        >>> gt_instances = BaseDataElement(
        ...     metainfo=dict(img_id=0, img_shape=(640, 640)),
        ...     bboxes=jt.rand((6, 4)), scores=jt.rand((6,)))
        >>> gt_instances.set_metainfo(dict(img_shape=(1280, 1280)))
        >>> gt_instances.img_shape  # (1280, 1280)
        >>> gt_instances.bboxes = gt_instances.bboxes * 2
        >>> gt_instances.get('img_shape', None)  # (1280, 1280)
        >>> gt_instances.get('bboxes', None)  # 6x4 tensor
        >>> del gt_instances.img_shape
        >>> del gt_instances.bboxes
        >>> assert 'img_shape' not in gt_instances
        >>> assert 'bboxes' not in gt_instances
        >>> gt_instances.pop('img_shape', None)  # None
        >>> gt_instances.pop('bboxes', None)  # None

        >>> # Tensor-like
        >>> cuda_instances = gt_instances.cuda()
        >>> cuda_instances = gt_instances.to('cuda:0')
        >>> cpu_instances = cuda_instances.cpu()
        >>> cpu_instances = cuda_instances.to('cpu')
        >>> fp16_instances = cuda_instances.to(
        ...     device=None, dtype=jt.float16, non_blocking=False,
        ...     copy=False, memory_format=jt.preserve_format)
        >>> cpu_instances = cuda_instances.detach()
        >>> np_instances = cpu_instances.numpy()

        >>> # print
        >>> metainfo = dict(img_shape=(800, 1196, 3))
        >>> gt_instances = BaseDataElement(
        ...     metainfo=metainfo, det_labels=jt.LongTensor([0, 1, 2, 3]))
        >>> sample = BaseDataElement(metainfo=metainfo,
        ...                          gt_instances=gt_instances)
        >>> print(sample)
        <BaseDataElement(
            META INFORMATION
            img_shape: (800, 1196, 3)
            DATA FIELDS
            gt_instances: <BaseDataElement(
                    META INFORMATION
                    img_shape: (800, 1196, 3)
                    DATA FIELDS
                    det_labels: tensor([0, 1, 2, 3])
                ) at 0x7f0ec5eadc70>
        ) at 0x7f0fea49e130>

        >>> # inheritance
        >>> class DetDataSample(BaseDataElement):
        ...     @property
        ...     def proposals(self):
        ...         return self._proposals
        ...     @proposals.setter
        ...     def proposals(self, value):
        ...         self.set_field(value, '_proposals', dtype=BaseDataElement)
        ...     @proposals.deleter
        ...     def proposals(self):
        ...         del self._proposals
        ...     @property
        ...     def gt_instances(self):
        ...         return self._gt_instances
        ...     @gt_instances.setter
        ...     def gt_instances(self, value):
        ...         self.set_field(value, '_gt_instances',
        ...                        dtype=BaseDataElement)
        ...     @gt_instances.deleter
        ...     def gt_instances(self):
        ...         del self._gt_instances
        ...     @property
        ...     def pred_instances(self):
        ...         return self._pred_instances
        ...     @pred_instances.setter
        ...     def pred_instances(self, value):
        ...         self.set_field(value, '_pred_instances',
        ...                        dtype=BaseDataElement)
        ...     @pred_instances.deleter
        ...     def pred_instances(self):
        ...         del self._pred_instances
        >>> det_sample = DetDataSample()
        >>> proposals = BaseDataElement(bboxes=jt.rand((5, 4)))
        >>> det_sample.proposals = proposals
        >>> assert 'proposals' in det_sample
        >>> assert det_sample.proposals == proposals
        >>> del det_sample.proposals
        >>> assert 'proposals' not in det_sample
        >>> with self.assertRaises(AssertionError):
        ...     det_sample.proposals = jt.rand((5, 4))
    """

    def __init__(self, *, metainfo: Optional[dict] = None, **kwargs) -> None:

        self._metainfo_fields: set = set()
        self._data_fields: set = set()

        if metainfo is not None:
            self.set_metainfo(metainfo=metainfo)
        if kwargs:
            self.set_data(kwargs)

    def set_metainfo(self, metainfo: dict) -> None:
        """Set or change key-value pairs in ``metainfo_field`` by parameter
        ``metainfo``.

        Args:
            metainfo (dict): A dict contains the meta information
                of image, such as ``img_shape``, ``scale_factor``, etc.
        """
        assert isinstance(
            metainfo,
            dict), f'metainfo should be a ``dict`` but got {type(metainfo)}'
        meta = copy.deepcopy(metainfo)
        for k, v in meta.items():
            self.set_field(name=k, value=v, field_type='metainfo', dtype=None)

    def set_data(self, data: dict) -> None:
        """Set or change key-value pairs in ``data_field`` by parameter
        ``data``.

        Args:
            data (dict): A dict contains annotations of image or
                model predictions.
        """
        assert isinstance(data,
                          dict), f'data should be a `dict` but got {data}'
        for k, v in data.items():
            # Use `setattr()` rather than `self.set_field` to allow `set_data`
            # to set property method.
            setattr(self, k, v)

    def update(self, instance: 'BaseDataElement') -> None:
        """The update() method updates the BaseDataElement with the elements
        from another BaseDataElement object.

        Args:
            instance (BaseDataElement): Another BaseDataElement object for
                update the current object.
        """
        assert isinstance(
            instance, BaseDataElement
        ), f'instance should be a `BaseDataElement` but got {type(instance)}'
        self.set_metainfo(dict(instance.metainfo_items()))
        self.set_data(dict(instance.items()))

    def new(self,
            *,
            metainfo: Optional[dict] = None,
            **kwargs) -> 'BaseDataElement':
        """Return a new data element with same type. If ``metainfo`` and
        ``data`` are None, the new data element will have same metainfo and
        data. If metainfo or data is not None, the new result will overwrite it
        with the input value.

        Args:
            metainfo (dict, optional): A dict contains the meta information
                of image, such as ``img_shape``, ``scale_factor``, etc.
                Defaults to None.
            kwargs (dict): A dict contains annotations of image or
                model predictions.

        Returns:
            BaseDataElement: A new data element with same type.
        """
        new_data = self.__class__()

        if metainfo is not None:
            new_data.set_metainfo(metainfo)
        else:
            new_data.set_metainfo(dict(self.metainfo_items()))
        if kwargs:
            new_data.set_data(kwargs)
        else:
            new_data.set_data(dict(self.items()))
        return new_data

    def clone(self):
        """Deep copy the current data element.

        Returns:
            BaseDataElement: The copy of current data element.
        """
        clone_data = self.__class__()
        clone_data.set_metainfo(dict(self.metainfo_items()))
        clone_data.set_data(dict(self.items()))
        return clone_data

    def keys(self) -> list:
        """
        Returns:
            list: Contains all keys in data_fields.
        """
        # We assume that the name of the attribute related to property is
        # '_' + the name of the property. We use this rule to filter out
        # private keys.
        # TODO: Use a more robust way to solve this problem
        private_keys = {
            '_' + key
            for key in self._data_fields
            if isinstance(getattr(type(self), key, None), property)
        }
        return list(self._data_fields - private_keys)

    def metainfo_keys(self) -> list:
        """
        Returns:
            list: Contains all keys in metainfo_fields.
        """
        return list(self._metainfo_fields)

    def values(self) -> list:
        """
        Returns:
            list: Contains all values in data.
        """
        return [getattr(self, k) for k in self.keys()]

    def metainfo_values(self) -> list:
        """
        Returns:
            list: Contains all values in metainfo.
        """
        return [getattr(self, k) for k in self.metainfo_keys()]

    def all_keys(self) -> list:
        """
        Returns:
            list: Contains all keys in metainfo and data.
        """
        return self.metainfo_keys() + self.keys()

    def all_values(self) -> list:
        """
        Returns:
            list: Contains all values in metainfo and data.
        """
        return self.metainfo_values() + self.values()

    def all_items(self) -> Iterator[Tuple[str, Any]]:
        """
        Returns:
            iterator: An iterator object whose element is (key, value) tuple
            pairs for ``metainfo`` and ``data``.
        """
        for k in self.all_keys():
            yield (k, getattr(self, k))

    def items(self) -> Iterator[Tuple[str, Any]]:
        """
        Returns:
            iterator: An iterator object whose element is (key, value) tuple
            pairs for ``data``.
        """
        for k in self.keys():
            yield (k, getattr(self, k))

    def metainfo_items(self) -> Iterator[Tuple[str, Any]]:
        """
        Returns:
            iterator: An iterator object whose element is (key, value) tuple
            pairs for ``metainfo``.
        """
        for k in self.metainfo_keys():
            yield (k, getattr(self, k))

    @property
    def metainfo(self) -> dict:
        """dict: A dict contains metainfo of current data element."""
        return dict(self.metainfo_items())

    def __setattr__(self, name: str, value: Any):
        """Setattr is only used to set data."""
        if name in ('_metainfo_fields', '_data_fields'):
            if not hasattr(self, name):
                super().__setattr__(name, value)
            else:
                raise AttributeError(f'{name} has been used as a '
                                     'private attribute, which is immutable.')
        else:
            self.set_field(
                name=name, value=value, field_type='data', dtype=None)

    def __delattr__(self, item: str):
        """Delete the item in dataelement.

        Args:
            item (str): The key to delete.
        """
        if item in ('_metainfo_fields', '_data_fields'):
            raise AttributeError(f'{item} has been used as a '
                                 'private attribute, which is immutable.')
        super().__delattr__(item)
        if item in self._metainfo_fields:
            self._metainfo_fields.remove(item)
        elif item in self._data_fields:
            self._data_fields.remove(item)

    # dict-like methods
    __delitem__ = __delattr__

    def get(self, key, default=None) -> Any:
        """Get property in data and metainfo as the same as python."""
        # Use `getattr()` rather than `self.__dict__.get()` to allow getting
        # properties.
        return getattr(self, key, default)

    def pop(self, *args) -> Any:
        """Pop property in data and metainfo as the same as python."""
        assert len(args) < 3, '``pop`` get more than 2 arguments'
        name = args[0]
        if name in self._metainfo_fields:
            self._metainfo_fields.remove(args[0])
            return self.__dict__.pop(*args)

        elif name in self._data_fields:
            self._data_fields.remove(args[0])
            return self.__dict__.pop(*args)

        # with default value
        elif len(args) == 2:
            return args[1]
        else:
            # don't just use 'self.__dict__.pop(*args)' for only popping key in
            # metainfo or data
            raise KeyError(f'{args[0]} is not contained in metainfo or data')

    def __contains__(self, item: str) -> bool:
        """Whether the item is in dataelement.

        Args:
            item (str): The key to inquire.
        """
        return item in self._data_fields or item in self._metainfo_fields

    def set_field(self,
                  value: Any,
                  name: str,
                  dtype: Optional[Union[Type, Tuple[Type, ...]]] = None,
                  field_type: str = 'data') -> None:
        """Special method for set union field, used as property.setter
        functions."""
        assert field_type in ['metainfo', 'data']
        if dtype is not None:
            assert isinstance(
                value,
                dtype), f'{value} should be a {dtype} but got {type(value)}'

        if field_type == 'metainfo':
            if name in self._data_fields:
                raise AttributeError(
                    f'Cannot set {name} to be a field of metainfo '
                    f'because {name} is already a data field')
            self._metainfo_fields.add(name)
        else:
            if name in self._metainfo_fields:
                raise AttributeError(
                    f'Cannot set {name} to be a field of data '
                    f'because {name} is already a metainfo field')
            self._data_fields.add(name)
        super().__setattr__(name, value)

    # Tensor-like methods
    def to(self, *args, **kwargs) -> 'BaseDataElement':
        """Apply same name function to all tensors in data_fields."""
        new_data = self.new()
        for k, v in self.items():
            if hasattr(v, 'to'):
                v = v.to(*args, **kwargs)
                data = {k: v}
                new_data.set_data(data)
        return new_data

    # Tensor-like methods
    def cpu(self) -> 'BaseDataElement':
        """Convert all tensors to CPU in data."""
        new_data = self.new()
        for k, v in self.items():
            if isinstance(v, (jt.Var, BaseDataElement)):
                v = v.cpu()
                data = {k: v}
                new_data.set_data(data)
        return new_data

    # Tensor-like methods
    def cuda(self) -> 'BaseDataElement':
        """Convert all tensors to GPU in data."""
        new_data = self.new()
        for k, v in self.items():
            if isinstance(v, (jt.Var, BaseDataElement)):
                v = v.cuda()
                data = {k: v}
                new_data.set_data(data)
        return new_data

    # Tensor-like methods
    def musa(self) -> 'BaseDataElement':
        """Convert all tensors to musa in data."""
        new_data = self.new()
        for k, v in self.items():
            if isinstance(v, (jt.Var, BaseDataElement)):
                v = v.musa()
                data = {k: v}
                new_data.set_data(data)
        return new_data

    # Tensor-like methods
    def npu(self) -> 'BaseDataElement':
        """Convert all tensors to NPU in data."""
        new_data = self.new()
        for k, v in self.items():
            if isinstance(v, (jt.Var, BaseDataElement)):
                v = v.npu()
                data = {k: v}
                new_data.set_data(data)
        return new_data

    def mlu(self) -> 'BaseDataElement':
        """Convert all tensors to MLU in data."""
        new_data = self.new()
        for k, v in self.items():
            if isinstance(v, (jt.Var, BaseDataElement)):
                v = v.mlu()
                data = {k: v}
                new_data.set_data(data)
        return new_data

    # Tensor-like methods
    def detach(self) -> 'BaseDataElement':
        """Detach all tensors in data."""
        new_data = self.new()
        for k, v in self.items():
            if isinstance(v, (jt.Var, BaseDataElement)):
                v = v.detach()
                data = {k: v}
                new_data.set_data(data)
        return new_data

    # Tensor-like methods
    def numpy(self) -> 'BaseDataElement':
        """Convert all tensors to np.ndarray in data."""
        new_data = self.new()
        for k, v in self.items():
            if isinstance(v, (jt.Var, BaseDataElement)):
                v = v.detach().cpu().numpy()
                data = {k: v}
                new_data.set_data(data)
        return new_data

    def to_tensor(self) -> 'BaseDataElement':
        """Convert all np.ndarray to tensor in data."""
        new_data = self.new()
        for k, v in self.items():
            data = {}
            if isinstance(v, np.ndarray):
                v = jt.from_numpy(v)
                data[k] = v
            elif isinstance(v, BaseDataElement):
                v = v.to_tensor()
                data[k] = v
            new_data.set_data(data)
        return new_data

    def to_dict(self) -> dict:
        """Convert BaseDataElement to dict."""
        return {
            k: v.to_dict() if isinstance(v, BaseDataElement) else v
            for k, v in self.all_items()
        }

    def __repr__(self) -> str:
        """Represent the object."""

        def _addindent(s_: str, num_spaces: int) -> str:
            """This func is modified from `pytorch` https://github.com/pytorch/
            pytorch/blob/b17b2b1cc7b017c3daaeff8cc7ec0f514d42ec37/torch/nn/modu
            les/module.py#L29.

            Args:
                s_ (str): The string to add spaces.
                num_spaces (int): The num of space to add.

            Returns:
                str: The string after add indent.
            """
            s = s_.split('\n')
            # don't do anything for single-line stuff
            if len(s) == 1:
                return s_
            first = s.pop(0)
            s = [(num_spaces * ' ') + line for line in s]
            s = '\n'.join(s)  # type: ignore
            s = first + '\n' + s  # type: ignore
            return s  # type: ignore

        def dump(obj: Any) -> str:
            """Represent the object.

            Args:
                obj (Any): The obj to represent.

            Returns:
                str: The represented str.
            """
            _repr = ''
            if isinstance(obj, dict):
                for k, v in obj.items():
                    _repr += f'\n{k}: {_addindent(dump(v), 4)}'
            elif isinstance(obj, BaseDataElement):
                _repr += '\n\n    META INFORMATION'
                metainfo_items = dict(obj.metainfo_items())
                _repr += _addindent(dump(metainfo_items), 4)
                _repr += '\n\n    DATA FIELDS'
                items = dict(obj.items())
                _repr += _addindent(dump(items), 4)
                classname = obj.__class__.__name__
                _repr = f'<{classname}({_repr}\n) at {hex(id(obj))}>'
            else:
                _repr += repr(obj)
            return _repr

        return dump(self)


# In[ ]:





# In[12]:


class InstanceData(BaseDataElement):
    """Data structure for instance-level annotations or predictions.

    Subclass of :class:`BaseDataElement`. All value in `data_fields`
    should have the same length. This design refer to
    https://github.com/facebookresearch/detectron2/blob/master/detectron2/structures/instances.py # noqa E501
    InstanceData also support extra functions: ``index``, ``slice`` and ``cat`` for data field. The type of value
    in data field can be base data structure such as `jt.Var`, `numpy.ndarray`, `list`, `str`, `tuple`,
    and can be customized data structure that has ``__len__``, ``__getitem__`` and ``cat`` attributes.

    Examples:
        >>> # custom data structure
        >>> class TmpObject:
        ...     def __init__(self, tmp) -> None:
        ...         assert isinstance(tmp, list)
        ...         self.tmp = tmp
        ...     def __len__(self):
        ...         return len(self.tmp)
        ...     def __getitem__(self, item):
        ...         if isinstance(item, int):
        ...             if item >= len(self) or item < -len(self):  # type:ignore
        ...                 raise IndexError(f'Index {item} out of range!')
        ...             else:
        ...                 # keep the dimension
        ...                 item = slice(item, None, len(self))
        ...         return TmpObject(self.tmp[item])
        ...     @staticmethod
        ...     def cat(tmp_objs):
        ...         assert all(isinstance(results, TmpObject) for results in tmp_objs)
        ...         if len(tmp_objs) == 1:
        ...             return tmp_objs[0]
        ...         tmp_list = [tmp_obj.tmp for tmp_obj in tmp_objs]
        ...         tmp_list = list(itertools.chain(*tmp_list))
        ...         new_data = TmpObject(tmp_list)
        ...         return new_data
        ...     def __repr__(self):
        ...         return str(self.tmp)
        >>> from mmengine.structures import InstanceData
        >>> import numpy as np
        >>> import torch
        >>> img_meta = dict(img_shape=(800, 1196, 3), pad_shape=(800, 1216, 3))
        >>> instance_data = InstanceData(metainfo=img_meta)
        >>> 'img_shape' in instance_data
        True
        >>> instance_data.det_labels = torch.LongTensor([2, 3])
        >>> instance_data["det_scores"] = jt.Var([0.8, 0.7])
        >>> instance_data.bboxes = torch.rand((2, 4))
        >>> instance_data.polygons = TmpObject([[1, 2, 3, 4], [5, 6, 7, 8]])
        >>> len(instance_data)
        2
        >>> print(instance_data)
        <InstanceData(
            META INFORMATION
            img_shape: (800, 1196, 3)
            pad_shape: (800, 1216, 3)
            DATA FIELDS
            det_labels: tensor([2, 3])
            det_scores: tensor([0.8000, 0.7000])
            bboxes: tensor([[0.4997, 0.7707, 0.0595, 0.4188],
                        [0.8101, 0.3105, 0.5123, 0.6263]])
            polygons: [[1, 2, 3, 4], [5, 6, 7, 8]]
        ) at 0x7fb492de6280>
        >>> sorted_results = instance_data[instance_data.det_scores.sort().indices]
        >>> sorted_results.det_scores
        tensor([0.7000, 0.8000])
        >>> print(instance_data[instance_data.det_scores > 0.75])
        <InstanceData(
            META INFORMATION
            img_shape: (800, 1196, 3)
            pad_shape: (800, 1216, 3)
            DATA FIELDS
            det_labels: tensor([2])
            det_scores: tensor([0.8000])
            bboxes: tensor([[0.4997, 0.7707, 0.0595, 0.4188]])
            polygons: [[1, 2, 3, 4]]
        ) at 0x7f64ecf0ec40>
        >>> print(instance_data[instance_data.det_scores > 1])
        <InstanceData(
            META INFORMATION
            img_shape: (800, 1196, 3)
            pad_shape: (800, 1216, 3)
            DATA FIELDS
            det_labels: tensor([], dtype=torch.int64)
            det_scores: tensor([])
            bboxes: tensor([], size=(0, 4))
            polygons: []
        ) at 0x7f660a6a7f70>
        >>> print(instance_data.cat([instance_data, instance_data]))
        <InstanceData(
            META INFORMATION
            img_shape: (800, 1196, 3)
            pad_shape: (800, 1216, 3)
            DATA FIELDS
            det_labels: tensor([2, 3, 2, 3])
            det_scores: tensor([0.8000, 0.7000, 0.8000, 0.7000])
            bboxes: tensor([[0.4997, 0.7707, 0.0595, 0.4188],
                        [0.8101, 0.3105, 0.5123, 0.6263],
                        [0.4997, 0.7707, 0.0595, 0.4188],
                        [0.8101, 0.3105, 0.5123, 0.6263]])
            polygons: [[1, 2, 3, 4], [5, 6, 7, 8], [1, 2, 3, 4], [5, 6, 7, 8]]
        ) at 0x7f203542feb0>
    """
    BoolTypeTensor = Union[jt.Var, jt.Var]
    LongTypeTensor = Union[jt.Var, jt.Var]

#     if get_device() == 'npu':
#         BoolTypeTensor = Union[torch.BoolTensor, torch.npu.BoolTensor]
#         LongTypeTensor = Union[torch.LongTensor, torch.npu.LongTensor]
#     elif get_device() == 'mlu':
#         BoolTypeTensor = Union[torch.BoolTensor, torch.mlu.BoolTensor]
#         LongTypeTensor = Union[torch.LongTensor, torch.mlu.LongTensor]
#     elif get_device() == 'musa':
#         BoolTypeTensor = Union[torch.BoolTensor, torch.musa.BoolTensor]
#         LongTypeTensor = Union[torch.LongTensor, torch.musa.LongTensor]
#     else:
#         BoolTypeTensor = Union[torch.BoolTensor, torch.cuda.BoolTensor]
#         LongTypeTensor = Union[torch.LongTensor, torch.cuda.LongTensor]

    IndexType: Union[Any] = Union[str, slice, int, list, LongTypeTensor,
                                  BoolTypeTensor, np.ndarray]
    def __setattr__(self, name: str, value: Sized):
        """Setattr is only used to set data.

        The value must have the attribute of `__len__` and have the same length
        of `InstanceData`.
        """
        if name in ('_metainfo_fields', '_data_fields'):
            if not hasattr(self, name):
                super().__setattr__(name, value)
            else:
                raise AttributeError(f'{name} has been used as a '
                                     'private attribute, which is immutable.')

        else:
            assert isinstance(value,
                              Sized), 'value must contain `__len__` attribute'

            if len(self) > 0:
                assert len(value) == len(self), 'The length of '                                                 f'values {len(value)} is '                                                 'not consistent with '                                                 'the length of this '                                                 ':obj:`InstanceData` '                                                 f'{len(self)}'
            super().__setattr__(name, value)

    __setitem__ = __setattr__

    def __getitem__(self, item: IndexType) -> 'InstanceData':
        """
        Args:
            item (str, int, list, :obj:`slice`, :obj:`numpy.ndarray`,
                :obj:`torch.LongTensor`, :obj:`torch.BoolTensor`):
                Get the corresponding values according to item.

        Returns:
            :obj:`InstanceData`: Corresponding values.
        """
        assert isinstance(item, IndexType.__args__)
        if isinstance(item, list):
            item = np.array(item)
        if isinstance(item, np.ndarray):
            # The default int type of numpy is platform dependent, int32 for
            # windows and int64 for linux. `jt.Var` requires the index
            # should be int64, therefore we simply convert it to int64 here.
            # More details in https://github.com/numpy/numpy/issues/9464
            item = item.astype(np.int64) if item.dtype == np.int32 else item
            item = torch.from_numpy(item)

        if isinstance(item, str):
            return getattr(self, item)

        if isinstance(item, int):
            if item >= len(self) or item < -len(self):  # type:ignore
                raise IndexError(f'Index {item} out of range!')
            else:
                # keep the dimension
                item = slice(item, None, len(self))

        new_data = self.__class__(metainfo=self.metainfo)
        if isinstance(item, jt.Var):
            assert item.dim() == 1, 'Only support to get the'                                     ' values along the first dimension.'
            if isinstance(item, BoolTypeTensor.__args__):
                assert len(item) == len(self), 'The shape of the '                                                'input(BoolTensor) '                                                f'{len(item)} '                                                'does not match the shape '                                                'of the indexed tensor '                                                'in results_field '                                                f'{len(self)} at '                                                'first dimension.'

            for k, v in self.items():
                if isinstance(v, jt.Var):
                    new_data[k] = v[item]
                elif isinstance(v, np.ndarray):
                    new_data[k] = v[item.cpu().numpy()]
                elif isinstance(
                        v, (str, list, tuple)) or (hasattr(v, '__getitem__')
                                                   and hasattr(v, 'cat')):
                    # convert to indexes from BoolTensor
                    if isinstance(item, BoolTypeTensor.__args__):
                        indexes = torch.nonzero(item).view(
                            -1).cpu().numpy().tolist()
                    else:
                        indexes = item.cpu().numpy().tolist()
                    slice_list = []
                    if indexes:
                        for index in indexes:
                            slice_list.append(slice(index, None, len(v)))
                    else:
                        slice_list.append(slice(None, 0, None))
                    r_list = [v[s] for s in slice_list]
                    if isinstance(v, (str, list, tuple)):
                        new_value = r_list[0]
                        for r in r_list[1:]:
                            new_value = new_value + r
                    else:
                        new_value = v.cat(r_list)
                    new_data[k] = new_value
                else:
                    raise ValueError(
                        f'The type of `{k}` is `{type(v)}`, which has no '
                        'attribute of `cat`, so it does not '
                        'support slice with `bool`')

        else:
            # item is a slice
            for k, v in self.items():
                new_data[k] = v[item]
        return new_data  # type:ignore

    @staticmethod
    def cat(instances_list: List['InstanceData']) -> 'InstanceData':
        """Concat the instances of all :obj:`InstanceData` in the list.

        Note: To ensure that cat returns as expected, make sure that
        all elements in the list must have exactly the same keys.

        Args:
            instances_list (list[:obj:`InstanceData`]): A list
                of :obj:`InstanceData`.

        Returns:
            :obj:`InstanceData`
        """
        assert all(
            isinstance(results, InstanceData) for results in instances_list)
        assert len(instances_list) > 0
        if len(instances_list) == 1:
            return instances_list[0]

        # metainfo and data_fields must be exactly the
        # same for each element to avoid exceptions.
        field_keys_list = [
            instances.all_keys() for instances in instances_list
        ]
        assert len({len(field_keys) for field_keys in field_keys_list})                == 1 and len(set(itertools.chain(*field_keys_list)))                == len(field_keys_list[0]), 'There are different keys in '                                            '`instances_list`, which may '                                            'cause the cat operation '                                            'to fail. Please make sure all '                                            'elements in `instances_list` '                                            'have the exact same key.'

        new_data = instances_list[0].__class__(
            metainfo=instances_list[0].metainfo)
        for k in instances_list[0].keys():
            values = [results[k] for results in instances_list]
            v0 = values[0]
            if isinstance(v0, jt.Var):
                new_values = torch.cat(values, dim=0)
            elif isinstance(v0, np.ndarray):
                new_values = np.concatenate(values, axis=0)
            elif isinstance(v0, (str, list, tuple)):
                new_values = v0[:]
                for v in values[1:]:
                    new_values += v
            elif hasattr(v0, 'cat'):
                new_values = v0.cat(values)
            else:
                raise ValueError(
                    f'The type of `{k}` is `{type(v0)}` which has no '
                    'attribute of `cat`')
            new_data[k] = new_values
        return new_data  # type:ignore

    def __len__(self) -> int:
        """int: The length of InstanceData."""
        if len(self._data_fields) > 0:
            return len(self.values()[0])
        else:
            return 0


# In[ ]:





# In[13]:


class PixelData(BaseDataElement):
    """Data structure for pixel-level annotations or predictions.

    All data items in ``data_fields`` of ``PixelData`` meet the following
    requirements:

    - They all have 3 dimensions in orders of channel, height, and width.
    - They should have the same height and width.

    Examples:
        >>> metainfo = dict(
        ...     img_id=random.randint(0, 100),
        ...     img_shape=(random.randint(400, 600), random.randint(400, 600)))
        >>> image = np.random.randint(0, 255, (4, 20, 40))
        >>> featmap = torch.randint(0, 255, (10, 20, 40))
        >>> pixel_data = PixelData(metainfo=metainfo,
        ...                        image=image,
        ...                        featmap=featmap)
        >>> print(pixel_data.shape)
        (20, 40)

        >>> # slice
        >>> slice_data = pixel_data[10:20, 20:40]
        >>> assert slice_data.shape == (10, 20)
        >>> slice_data = pixel_data[10, 20]
        >>> assert slice_data.shape == (1, 1)

        >>> # set
        >>> pixel_data.map3 = torch.randint(0, 255, (20, 40))
        >>> assert tuple(pixel_data.map3.shape) == (1, 20, 40)
        >>> with self.assertRaises(AssertionError):
        ...     # The dimension must be 3 or 2
        ...     pixel_data.map2 = torch.randint(0, 255, (1, 3, 20, 40))
    """

    def __setattr__(self, name: str, value: Union[jt.Var, np.ndarray]):
        """Set attributes of ``PixelData``.

        If the dimension of value is 2 and its shape meet the demand, it
        will automatically expand its channel-dimension.

        Args:
            name (str): The key to access the value, stored in `PixelData`.
            value (Union[jt.Var, np.ndarray]): The value to store in.
                The type of value must be `jt.Var` or `np.ndarray`,
                and its shape must meet the requirements of `PixelData`.
        """
        if name in ('_metainfo_fields', '_data_fields'):
            if not hasattr(self, name):
                super().__setattr__(name, value)
            else:
                raise AttributeError(f'{name} has been used as a '
                                     'private attribute, which is immutable.')

        else:
            assert isinstance(value, (jt.Var, np.ndarray)),                 f'Can not set {type(value)}, only support'                 f' {(jt.Var, np.ndarray)}'

            if self.shape:
                assert tuple(value.shape[-2:]) == self.shape, (
                    'The height and width of '
                    f'values {tuple(value.shape[-2:])} is '
                    'not consistent with '
                    'the shape of this '
                    ':obj:`PixelData` '
                    f'{self.shape}')
            assert value.ndim in [
                2, 3
            ], f'The dim of value must be 2 or 3, but got {value.ndim}'
            if value.ndim == 2:
                value = value[None]
                warnings.warn('The shape of value will convert from '
                              f'{value.shape[-2:]} to {value.shape}')
            super().__setattr__(name, value)

    # TODO torch.Long/bool
    def __getitem__(self, item: Sequence[Union[int, slice]]) -> 'PixelData':
        """
        Args:
            item (Sequence[Union[int, slice]]): Get the corresponding values
                according to item.

        Returns:
            :obj:`PixelData`: Corresponding values.
        """

        new_data = self.__class__(metainfo=self.metainfo)
        if isinstance(item, tuple):

            assert len(item) == 2, 'Only support to slice height and width'
            tmp_item: List[slice] = list()
            for index, single_item in enumerate(item[::-1]):
                if isinstance(single_item, int):
                    tmp_item.insert(
                        0, slice(single_item, None, self.shape[-index - 1]))
                elif isinstance(single_item, slice):
                    tmp_item.insert(0, single_item)
                else:
                    raise TypeError(
                        'The type of element in input must be int or slice, '
                        f'but got {type(single_item)}')
            tmp_item.insert(0, slice(None, None, None))
            item = tuple(tmp_item)
            for k, v in self.items():
                setattr(new_data, k, v[item])
        else:
            raise TypeError(
                f'Unsupported type {type(item)} for slicing PixelData')
        return new_data

    @property
    def shape(self):
        """The shape of pixel data."""
        if len(self._data_fields) > 0:
            return tuple(self.values()[0].shape[-2:])
        else:
            return None

    # TODO padding, resize


# In[ ]:





# In[14]:


class DetDataSample(BaseDataElement):
    @property
    def proposals(self) -> InstanceData:
        return self._proposals

    @proposals.setter
    def proposals(self, value: InstanceData):
        self.set_field(value, '_proposals', dtype=InstanceData)

    @proposals.deleter
    def proposals(self):
        del self._proposals

    @property
    def gt_instances(self) -> InstanceData:
        return self._gt_instances

    @gt_instances.setter
    def gt_instances(self, value: InstanceData):
        self.set_field(value, '_gt_instances', dtype=InstanceData)

    @gt_instances.deleter
    def gt_instances(self):
        del self._gt_instances

    @property
    def pred_instances(self) -> InstanceData:
        return self._pred_instances

    @pred_instances.setter
    def pred_instances(self, value: InstanceData):
        self.set_field(value, '_pred_instances', dtype=InstanceData)

    @pred_instances.deleter
    def pred_instances(self):
        del self._pred_instances

    # directly add ``pred_track_instances`` in ``DetDataSample``
    # so that the ``TrackDataSample`` does not bother to access the
    # instance-level information.
    @property
    def pred_track_instances(self) -> InstanceData:
        return self._pred_track_instances

    @pred_track_instances.setter
    def pred_track_instances(self, value: InstanceData):
        self.set_field(value, '_pred_track_instances', dtype=InstanceData)

    @pred_track_instances.deleter
    def pred_track_instances(self):
        del self._pred_track_instances

    @property
    def ignored_instances(self) -> InstanceData:
        return self._ignored_instances

    @ignored_instances.setter
    def ignored_instances(self, value: InstanceData):
        self.set_field(value, '_ignored_instances', dtype=InstanceData)

    @ignored_instances.deleter
    def ignored_instances(self):
        del self._ignored_instances

    @property
    def gt_panoptic_seg(self) -> PixelData:
        return self._gt_panoptic_seg

    @gt_panoptic_seg.setter
    def gt_panoptic_seg(self, value: PixelData):
        self.set_field(value, '_gt_panoptic_seg', dtype=PixelData)

    @gt_panoptic_seg.deleter
    def gt_panoptic_seg(self):
        del self._gt_panoptic_seg

    @property
    def pred_panoptic_seg(self) -> PixelData:
        return self._pred_panoptic_seg

    @pred_panoptic_seg.setter
    def pred_panoptic_seg(self, value: PixelData):
        self.set_field(value, '_pred_panoptic_seg', dtype=PixelData)

    @pred_panoptic_seg.deleter
    def pred_panoptic_seg(self):
        del self._pred_panoptic_seg

    @property
    def gt_sem_seg(self) -> PixelData:
        return self._gt_sem_seg

    @gt_sem_seg.setter
    def gt_sem_seg(self, value: PixelData):
        self.set_field(value, '_gt_sem_seg', dtype=PixelData)

    @gt_sem_seg.deleter
    def gt_sem_seg(self):
        del self._gt_sem_seg

    @property
    def pred_sem_seg(self) -> PixelData:
        return self._pred_sem_seg

    @pred_sem_seg.setter
    def pred_sem_seg(self, value: PixelData):
        self.set_field(value, '_pred_sem_seg', dtype=PixelData)

    @pred_sem_seg.deleter
    def pred_sem_seg(self):
        del self._pred_sem_seg


# In[ ]:





# In[15]:


class PackDetInputs:
    """Pack the inputs data for the detection / semantic segmentation /
    panoptic segmentation.

    The ``img_meta`` item is always populated.  The contents of the
    ``img_meta`` dictionary depends on ``meta_keys``. By default this includes:

        - ``img_id``: id of the image

        - ``img_path``: path to the image file

        - ``ori_shape``: original shape of the image as a tuple (h, w)

        - ``img_shape``: shape of the image input to the network as a tuple \
            (h, w).  Note that images may be zero padded on the \
            bottom/right if the batch tensor is larger than this shape.

        - ``scale_factor``: a float indicating the preprocessing scale

        - ``flip``: a boolean indicating if image flip transform was used

        - ``flip_direction``: the flipping direction

    Args:
        meta_keys (Sequence[str], optional): Meta keys to be converted to
            ``mmcv.DataContainer`` and collected in ``data[img_metas]``.
            Default: ``('img_id', 'img_path', 'ori_shape', 'img_shape',
            'scale_factor', 'flip', 'flip_direction')``
    """
    mapping_table = {
        'gt_bboxes': 'bboxes',
        'gt_bboxes_labels': 'labels',
        'gt_masks': 'masks'
    }

    def __init__(self,
                 meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                            'scale_factor', 'flip', 'flip_direction')):
        self.meta_keys = meta_keys

    def __call__(self, results: dict) -> dict:
        """Method to pack the input data.

        Args:
            results (dict): Result dict from the data pipeline.

        Returns:
            dict:

            - 'inputs' (obj:`jt.Var`): The forward data of models.
            - 'data_sample' (obj:`DetDataSample`): The annotation info of the
                sample.
        """
        packed_results = dict()
        if 'img' in results:
            img = results['img']
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)
            # To improve the computational speed by by 3-5 times, apply:
            # If image is not contiguous, use
            # `numpy.transpose()` followed by `numpy.ascontiguousarray()`
            # If image is already contiguous, use
            # `torch.permute()` followed by `torch.contiguous()`
            # Refer to https://github.com/open-mmlab/mmdetection/pull/9533
            # for more details
            if not img.flags.c_contiguous:
                img = np.ascontiguousarray(img.transpose(2, 0, 1))
                img = to_tensor(img)
            else:
                img = to_tensor(img).permute(2, 0, 1).contiguous()

            packed_results['inputs'] = img

        if 'gt_ignore_flags' in results:
            valid_idx = np.where(results['gt_ignore_flags'] == 0)[0]
            ignore_idx = np.where(results['gt_ignore_flags'] == 1)[0]

        data_sample = DetDataSample()
        instance_data = InstanceData()
        ignore_instance_data = InstanceData()

        for key in self.mapping_table.keys():
            if key not in results:
                continue
            if key == 'gt_masks' or isinstance(results[key], BaseBoxes):
                if 'gt_ignore_flags' in results:
                    instance_data[
                        self.mapping_table[key]] = results[key][valid_idx]
                    ignore_instance_data[
                        self.mapping_table[key]] = results[key][ignore_idx]
                else:
                    instance_data[self.mapping_table[key]] = results[key]
            else:
                if 'gt_ignore_flags' in results:
                    instance_data[self.mapping_table[key]] = to_tensor(
                        results[key][valid_idx])
                    ignore_instance_data[self.mapping_table[key]] = to_tensor(
                        results[key][ignore_idx])
                else:
                    instance_data[self.mapping_table[key]] = to_tensor(
                        results[key])
        data_sample.gt_instances = instance_data
        data_sample.ignored_instances = ignore_instance_data

        if 'proposals' in results:
            proposals = InstanceData(
                bboxes=to_tensor(results['proposals']),
                scores=to_tensor(results['proposals_scores']))
            data_sample.proposals = proposals

        if 'gt_seg_map' in results:
            gt_sem_seg_data = dict(
                sem_seg=to_tensor(results['gt_seg_map'][None, ...].copy()))
            gt_sem_seg_data = PixelData(**gt_sem_seg_data)
            if 'ignore_index' in results:
                metainfo = dict(ignore_index=results['ignore_index'])
                gt_sem_seg_data.set_metainfo(metainfo)
            data_sample.gt_sem_seg = gt_sem_seg_data

        img_meta = {}
        for key in self.meta_keys:
            if key in results:
                img_meta[key] = results[key]
        data_sample.set_metainfo(img_meta)
        packed_results['data_samples'] = data_sample

        return packed_results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(meta_keys={self.meta_keys})'
        return repr_str


# In[ ]:





# In[16]:


class Base_RandomFlip:
    def __init__(self,
                 prob: Optional[Union[float, Iterable[float]]] = None,
                 direction: Union[str, Sequence[Optional[str]]] = 'horizontal',
                 swap_seg_labels: Optional[Sequence] = None) -> None:
        if isinstance(prob, list):
#             assert mmengine.is_list_of(prob, float)
            assert all(isinstance(pr,float) for pr in prob)
            assert 0 <= sum(prob) <= 1
        elif isinstance(prob, float):
            assert 0 <= prob <= 1
        else:
            raise ValueError(f'probs must be float or list of float, but                               got `{type(prob)}`.')
        self.prob = prob
        self.swap_seg_labels = swap_seg_labels

        valid_directions = ['horizontal', 'vertical', 'diagonal']
        if isinstance(direction, str):
            assert direction in valid_directions
        elif isinstance(direction, list):
#             assert mmengine.is_list_of(direction, str)
            assert all(isinstance(dr,str) for dr in direction)
            assert set(direction).issubset(set(valid_directions))
        else:
            raise ValueError(f'direction must be either str or list of str,                                but got `{type(direction)}`.')
        self.direction = direction

        if isinstance(prob, list):
            assert len(prob) == len(self.direction)

    def _flip_bbox(self, bboxes: np.ndarray, img_shape: Tuple[int, int],
                   direction: str) -> np.ndarray:
        assert bboxes.shape[-1] % 4 == 0
        flipped = bboxes.copy()
        h, w = img_shape
        if direction == 'horizontal':
            flipped[..., 0::4] = w - bboxes[..., 2::4]
            flipped[..., 2::4] = w - bboxes[..., 0::4]
        elif direction == 'vertical':
            flipped[..., 1::4] = h - bboxes[..., 3::4]
            flipped[..., 3::4] = h - bboxes[..., 1::4]
        elif direction == 'diagonal':
            flipped[..., 0::4] = w - bboxes[..., 2::4]
            flipped[..., 1::4] = h - bboxes[..., 3::4]
            flipped[..., 2::4] = w - bboxes[..., 0::4]
            flipped[..., 3::4] = h - bboxes[..., 1::4]
        else:
            raise ValueError(
                f"Flipping direction must be 'horizontal', 'vertical', \
                  or 'diagonal', but got '{direction}'")
        return flipped

    def _flip_keypoints(
        self,
        keypoints: np.ndarray,
        img_shape: Tuple[int, int],
        direction: str,
    ) -> np.ndarray:

        meta_info = keypoints[..., 2:]
        keypoints = keypoints[..., :2]
        flipped = keypoints.copy()
        h, w = img_shape
        if direction == 'horizontal':
            flipped[..., 0::2] = w - keypoints[..., 0::2]
        elif direction == 'vertical':
            flipped[..., 1::2] = h - keypoints[..., 1::2]
        elif direction == 'diagonal':
            flipped[..., 0::2] = w - keypoints[..., 0::2]
            flipped[..., 1::2] = h - keypoints[..., 1::2]
        else:
            raise ValueError(
                f"Flipping direction must be 'horizontal', 'vertical', \
                  or 'diagonal', but got '{direction}'")
        flipped = np.concatenate([flipped, meta_info], axis=-1)
        return flipped

    def _flip_seg_map(self, seg_map: dict, direction: str) -> np.ndarray:
        seg_map = imflip(seg_map, direction=direction)
        if self.swap_seg_labels is not None:
            # to handle datasets with left/right annotations
            # like 'Left-arm' and 'Right-arm' in LIP dataset
            # Modified from https://github.com/openseg-group/openseg.pytorch/blob/master/lib/datasets/tools/cv2_aug_transforms.py # noqa:E501
            # Licensed under MIT license
            temp = seg_map.copy()
            assert isinstance(self.swap_seg_labels, (tuple, list))
            for pair in self.swap_seg_labels:
                assert isinstance(pair, (tuple, list)) and len(pair) == 2,                     'swap_seg_labels must be a sequence with pair, but got '                     f'{self.swap_seg_labels}.'
                seg_map[temp == pair[0]] = pair[1]
                seg_map[temp == pair[1]] = pair[0]
        return seg_map

    def _choose_direction(self) -> str:
        """Choose the flip direction according to `prob` and `direction`"""
        if isinstance(self.direction,
                      Sequence) and not isinstance(self.direction, str):
            # None means non-flip
            direction_list: list = list(self.direction) + [None]
        elif isinstance(self.direction, str):
            # None means non-flip
            direction_list = [self.direction, None]

        if isinstance(self.prob, list):
            non_prob: float = 1 - sum(self.prob)
            prob_list = self.prob + [non_prob]
        elif isinstance(self.prob, float):
            non_prob = 1. - self.prob
            # exclude non-flip
            single_ratio = self.prob / (len(direction_list) - 1)
            prob_list = [single_ratio] * (len(direction_list) - 1) + [non_prob]

        cur_dir = np.random.choice(direction_list, p=prob_list)

        return cur_dir

    def _flip(self, results: dict) -> None:
        # flip image
        results['img'] = imflip(
            results['img'], direction=results['flip_direction'])

        img_shape = results['img'].shape[:2]

        # flip bboxes
        if results.get('gt_bboxes', None) is not None:
            results['gt_bboxes'] = self._flip_bbox(results['gt_bboxes'],
                                                   img_shape,
                                                   results['flip_direction'])

        # flip keypoints
        if results.get('gt_keypoints', None) is not None:
            results['gt_keypoints'] = self._flip_keypoints(
                results['gt_keypoints'], img_shape, results['flip_direction'])

        # flip seg map
        if results.get('gt_seg_map', None) is not None:
            results['gt_seg_map'] = self._flip_seg_map(
                results['gt_seg_map'], direction=results['flip_direction'])
            results['swap_seg_labels'] = self.swap_seg_labels

    def _flip_on_direction(self, results: dict) -> None:
        cur_dir = self._choose_direction()
        if cur_dir is None:
            results['flip'] = False
            results['flip_direction'] = None
        else:
            results['flip'] = True
            results['flip_direction'] = cur_dir
            self._flip(results)

    def __call__(self, results: dict) -> dict:
        
        self._flip_on_direction(results)

        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(prob={self.prob}, '
        repr_str += f'direction={self.direction})'

        return repr_str


# In[ ]:





# In[17]:


class RandomFlip(Base_RandomFlip):
    """Flip the image & bbox & mask & segmentation map. Added or Updated keys:
    flip, flip_direction, img, gt_bboxes, and gt_seg_map. There are 3 flip
    modes:

     - ``prob`` is float, ``direction`` is string: the image will be
         ``direction``ly flipped with probability of ``prob`` .
         E.g., ``prob=0.5``, ``direction='horizontal'``,
         then image will be horizontally flipped with probability of 0.5.
     - ``prob`` is float, ``direction`` is list of string: the image will
         be ``direction[i]``ly flipped with probability of
         ``prob/len(direction)``.
         E.g., ``prob=0.5``, ``direction=['horizontal', 'vertical']``,
         then image will be horizontally flipped with probability of 0.25,
         vertically with probability of 0.25.
     - ``prob`` is list of float, ``direction`` is list of string:
         given ``len(prob) == len(direction)``, the image will
         be ``direction[i]``ly flipped with probability of ``prob[i]``.
         E.g., ``prob=[0.3, 0.5]``, ``direction=['horizontal',
         'vertical']``, then image will be horizontally flipped with
         probability of 0.3, vertically with probability of 0.5.


    Required Keys:

    - img
    - gt_bboxes (BaseBoxes[torch.float32]) (optional)
    - gt_masks (BitmapMasks | PolygonMasks) (optional)
    - gt_seg_map (np.uint8) (optional)

    Modified Keys:

    - img
    - gt_bboxes
    - gt_masks
    - gt_seg_map

    Added Keys:

    - flip
    - flip_direction
    - homography_matrix


    Args:
         prob (float | list[float], optional): The flipping probability.
             Defaults to None.
         direction(str | list[str]): The flipping direction. Options
             If input is a list, the length must equal ``prob``. Each
             element in ``prob`` indicates the flip probability of
             corresponding direction. Defaults to 'horizontal'.
    """
    def _record_homography_matrix(self, results: dict) -> None:
        """Record the homography matrix for the RandomFlip."""
        cur_dir = results['flip_direction']
        h, w = results['img'].shape[:2]

        if cur_dir == 'horizontal':
            homography_matrix = np.array([[-1, 0, w], [0, 1, 0], [0, 0, 1]],
                                         dtype=np.float32)
        elif cur_dir == 'vertical':
            homography_matrix = np.array([[1, 0, 0], [0, -1, h], [0, 0, 1]],
                                         dtype=np.float32)
        elif cur_dir == 'diagonal':
            homography_matrix = np.array([[-1, 0, w], [0, -1, h], [0, 0, 1]],
                                         dtype=np.float32)
        else:
            homography_matrix = np.eye(3, dtype=np.float32)

        if results.get('homography_matrix', None) is None:
            results['homography_matrix'] = homography_matrix
        else:
            results['homography_matrix'] = homography_matrix @ results[
                'homography_matrix']

    def _flip(self, results: dict) -> None:
        """Flip images, bounding boxes, and semantic segmentation map."""
        # flip image
        results['img'] = imflip(
            results['img'], direction=results['flip_direction'])

        img_shape = results['img'].shape[:2]

        # flip bboxes
        if results.get('gt_bboxes', None) is not None:
            results['gt_bboxes'].flip_(img_shape, results['flip_direction'])

        # flip masks
        if results.get('gt_masks', None) is not None:
            results['gt_masks'] = results['gt_masks'].flip(
                results['flip_direction'])

        # flip segs
        if results.get('gt_seg_map', None) is not None:
            results['gt_seg_map'] = imflip(
                results['gt_seg_map'], direction=results['flip_direction'])

        # record homography matrix for flip
        self._record_homography_matrix(results)


# In[ ]:





# In[18]:


class Blur:
    def __init__(self,
                blur_limit= (3, 7),
                p: float = 0.5,
                always_apply = False):
        self.blur_limit = cast("tuple[int, int]", blur_limit)
        self.p = p
        self.always_apply = always_apply
        self._additional_targets: Dict[str, str] = {}

        # replay mode params
        self.deterministic = False
        self.save_key = "replay"
        self.params: Dict[Any, Any] = {}
        self.replay_mode = False
        self.applied_in_replay = False
    def __call__(self, img: np.ndarray, kernel: int, **params: Any) -> np.ndarray:
        """Apply blur to the input image.

        Args:
            img (np.ndarray): Image to blur.
            kernel (int): Size of the kernel for blur.
            **params (Any): Additional parameters.

        Returns:
            np.ndarray: Blurred image.

        """
        if self.random():
            return blur(img, kernel)
        else:
            return img
    def get_params(self) -> Dict[str, Any]:
        """Get parameters for the transform.

        Returns:
            dict[str, Any]: Dictionary with parameters.

        """
        kernel = sample_odd_from_range(
            self.py_random,
            self.blur_limit[0],
            self.blur_limit[1],
        )
        return {"kernel": kernel}
    def random(self):
        return random.random() < self.p


# In[ ]:





# In[19]:


class MedianBlur(Blur):
    def __init__(
        self,
        blur_limit= (3, 7),
        p: float = 0.5,
    ):
        super().__init__(blur_limit=blur_limit, p=p)
    def apply(self, img: np.ndarray, kernel: int, **params: Any) -> np.ndarray:
        """Apply median blur to the input image.

        Args:
            img (np.ndarray): Image to blur.
            kernel (int): Size of the kernel for blur.
            **params (Any): Additional parameters.

        Returns:
            np.ndarray: Median blurred image.

        """
        return median_blur(img, kernel) if self.random() else img


# In[ ]:





# In[20]:


class ToGray:
    def __init__(
        self,
        num_output_channels: int = 3,
        method: Literal[
            "weighted_average",
            "from_lab",
            "desaturation",
            "average",
            "max",
            "pca",
        ] = "weighted_average",
        p: float = 0.5,
    ):
        self.p = p
        self.num_output_channels = num_output_channels
        self.method = method
    def random(self):
        return random.random() < self.p
    def __call__(self, img: np.ndarray, **params: Any) -> np.ndarray:
        """Apply the ToGray transform to the input image.

        Args:
            img (np.ndarray): The input image to apply the ToGray transform to.
            **params (Any): Additional parameters (not used in this transform).

        Returns:
            np.ndarray: The image with the applied ToGray transform.

        """
        return to_gray(img, self.num_output_channels, self.method) if self.random() else img
#         if is_grayscale_image(img):
#             warnings.warn("The image is already gray.", stacklevel=2)
#             return img
#         if img.ndim == 2:
#             return img
#         elif img.ndim == 3 and img[2] == 3:
#             num_channels = get_num_channels(img)

#             if num_channels != NUM_RGB_CHANNELS and self.method not in {
#                 "desaturation",
#                 "average",
#                 "max",
#                 "pca",
#             }:
#                 msg = "ToGray transformation expects 3-channel images."
#                 raise TypeError(msg)

#             return to_gray(img, self.num_output_channels, self.method)
#         elif img.ndim == 3:
#             return img
#         else:
            
#     def batch(self, images: np.ndarray, **params: Any) -> np.ndarray:
#         """Apply ToGray to a batch of images.

#         Args:
#             images (np.ndarray): Batch of images with shape (N, H, W, C) or (N, H, W).
#             **params (Any): Additional parameters.

#         Returns:
#             np.ndarray: Batch of grayscale images.

#         """
# #         if is_grayscale_image(images, has_batch_dim=True):
# #             warnings.warn("The image is already gray.", stacklevel=2)
# #             return images

#         return to_gray(images, self.num_output_channels, self.method)


# In[ ]:





# In[21]:


class CLAHE:
    def __init__(
        self,
        clip_limit= 4.0,
        tile_grid_size: Tuple[int, int] = (8, 8),
        p: float = 0.5,
    ):
        self.p = p
        self.clip_limit = cast("tuple[float, float]", clip_limit)
        self.tile_grid_size = tile_grid_size
        self.random = random.random()
    def __call__(self, img: np.ndarray, **params: Any) -> np.ndarray:
        return clahe(img, clip_limit, self.tile_grid_size) if self.random < self.p else img


# In[ ]:





# In[22]:


class BboxParams:
    """Parameters for bounding box transforms.

    Args:
        format (Literal["coco", "pascal_voc", "albumentations", "YOLO"]): Format of bounding boxes.
            Should be one of:
            - 'coco': [x_min, y_min, width, height], e.g. [97, 12, 150, 200].
            - 'pascal_voc': [x_min, y_min, x_max, y_max], e.g. [97, 12, 247, 212].
            - 'albumentations': like pascal_voc but normalized in [0, 1] range, e.g. [0.2, 0.3, 0.4, 0.5].
            - 'YOLO': [x_center, y_center, width, height] normalized in [0, 1] range, e.g. [0.1, 0.2, 0.3, 0.4].

        label_fields (Sequence[str] | None): List of fields that are joined with boxes,
            e.g., ['class_labels', 'scores']. Default: None.

        min_area (float): Minimum area of a bounding box. All bounding boxes whose visible area in pixels is less than
            this value will be removed. Default: 0.0.

        min_visibility (float): Minimum fraction of area for a bounding box to remain this box in the result.
            Should be in [0.0, 1.0] range. Default: 0.0.

        min_width (float): Minimum width of a bounding box in pixels or normalized units. Bounding boxes with width
            less than this value will be removed. Default: 0.0.

        min_height (float): Minimum height of a bounding box in pixels or normalized units. Bounding boxes with height
            less than this value will be removed. Default: 0.0.

        check_each_transform (bool): If True, performs checks for each dual transform. Default: True.

        clip (bool): If True, clips bounding boxes to image boundaries before applying any transform. Default: False.

        filter_invalid_bboxes (bool): If True, filters out invalid bounding boxes (e.g., boxes with negative dimensions
            or boxes where x_max < x_min or y_max < y_min) at the beginning of the pipeline. If clip=True, filtering
            is applied after clipping. Default: False.

        max_accept_ratio (float | None): Maximum allowed aspect ratio for bounding boxes. The aspect ratio is calculated
            as max(width/height, height/width), so it's always >= 1. Boxes with aspect ratio greater than this value
            will be filtered out. For example, if max_accept_ratio=3.0, boxes with width:height or height:width ratios
            greater than 3:1 will be removed. Set to None to disable aspect ratio filtering. Default: None.


    Note:
        The processing order for bounding boxes is:
        1. Convert to albumentations format (normalized pascal_voc)
        2. Clip boxes to image boundaries (if clip=True)
        3. Filter invalid boxes (if filter_invalid_bboxes=True)
        4. Apply transformations
        5. Filter boxes based on min_area, min_visibility, min_width, min_height
        6. Convert back to the original format

    Examples:
        >>> # Create BboxParams for COCO format with class labels
        >>> bbox_params = BboxParams(
        ...     format='coco',
        ...     label_fields=['class_labels'],
        ...     min_area=1024,
        ...     min_visibility=0.1
        ... )

        >>> # Create BboxParams that clips and filters invalid boxes
        >>> bbox_params = BboxParams(
        ...     format='pascal_voc',
        ...     clip=True,
        ...     filter_invalid_bboxes=True
        ... )
        >>> # Create BboxParams that filters extremely elongated boxes
        >>> bbox_params = BboxParams(
        ...     format='YOLO',
        ...     max_accept_ratio=5.0,  # Filter boxes with aspect ratio > 5:1
        ...     clip=True
        ... )

    """

    def __init__(
        self,
        format: Literal["coco", "pascal_voc", "albumentations", "yolo"],  # noqa: A002
        label_fields: Union[Sequence[Any], None] = None,
        min_area: float = 0.0,
        min_visibility: float = 0.0,
        min_width: float = 0.0,
        min_height: float = 0.0,
        check_each_transform: bool = True,
        clip: bool = False,
        filter_invalid_bboxes: bool = False,
        max_accept_ratio: Union[float, None] = None,
    ):
        self.format = format
        self.label_fields = label_fields
        self.min_area = min_area
        self.min_visibility = min_visibility
        self.min_width = min_width
        self.min_height = min_height
        self.check_each_transform = check_each_transform
        self.clip = clip
        self.filter_invalid_bboxes = filter_invalid_bboxes
        if max_accept_ratio is not None and max_accept_ratio < 1.0:
            raise ValueError(
                "max_accept_ratio must be >= 1.0 when provided, as aspect ratio is calculated as max(w/h, h/w)",
            )
        self.max_accept_ratio = max_accept_ratio  # e.g., 5.0

    def to_dict_private(self) -> Dict[str, Any]:
        """Get the private dictionary representation of bounding box parameters.

        Returns:
            dict[str, Any]: Dictionary containing the bounding box parameters.

        """
        data = {"format": self.format, "label_fields": self.label_fields}
        data.update(
            {
                "min_area": self.min_area,
                "min_visibility": self.min_visibility,
                "min_width": self.min_width,
                "min_height": self.min_height,
                "check_each_transform": self.check_each_transform,
                "clip": self.clip,
                "max_accept_ratio": self.max_accept_ratio,
            },
        )
        return data

    @classmethod
    def is_serializable(cls) -> bool:
        """Check if the bounding box parameters are serializable.

        Returns:
            bool: Always returns True as BboxParams is serializable.

        """
        return True

    @classmethod
    def get_class_fullname(cls) -> str:
        """Get the full name of the class.

        Returns:
            str: The string "BboxParams".

        """
        return "BboxParams"

    def __repr__(self) -> str:
        return (
            f"BboxParams(format={self.format}, label_fields={self.label_fields}, min_area={self.min_area},"
            f" min_visibility={self.min_visibility}, min_width={self.min_width}, min_height={self.min_height},"
            f" check_each_transform={self.check_each_transform}, clip={self.clip})"
        )


# In[ ]:





# In[23]:


class EMA(Module):
    def __init__(
        self,
        model: Module,
        momentum: float = 0.0002,
        gamma: int = 2000,
        interval=1,
        update_buffers: bool = False
    ):
        super().__init__()
#         self.module = deepcopy(model).need_grad_(False)
        self.module = module.clone().need_grad_(False)
        self.interval = interval
        self.register_buffer('steps',
                             jt.Var(0))
        self.update_buffers = update_buffers
        if update_buffers:
            self.avg_parameters = self.module.state_dict()
        else:
            self.avg_parameters = dict(self.module.named_parameters())
    def execute(self, *args, **kwargs):
        """Forward method of the averaged model."""
        return self.module(*args, **kwargs)
    def update_parameters(self, model: Module) -> None:
        """Update the parameters of the model. This method will execute the
        ``avg_func`` to compute the new parameters and update the model's
        parameters.

        Args:
            model (nn.Module): The model whose parameters will be averaged.
        """
        src_parameters = (
            dict(model.named_parameters()))
        if self.steps == 0:
            for k, p_avg in self.avg_parameters.items():
                p_avg.data.copy_(src_parameters[k].data)
        elif self.steps % self.interval == 0:
            for k, p_avg in self.avg_parameters.items():
                if p_avg.dtype.is_floating_point:
                    device = p_avg.device
                    self.avg_func(p_avg.data,
                                  src_parameters[k].data.to(device),
                                  self.steps)

        if not self.update_buffers:
            # If not update the buffers,
            # keep the buffers in sync with the source model.
            for b_avg, b_src in zip(self.module.buffers(), model.buffers()):
                b_avg.data.copy_(b_src.data.to(b_avg.device))
        self.steps += 1
    def avg_func(self, averaged_param: jt.Var, source_param: jt.Var,
                 steps: int) -> None:
        """Compute the moving average of the parameters using the exponential
        momentum strategy.

        Args:
            averaged_param (jt.Var): The averaged parameters.
            source_param (jt.Var): The source parameters.
            steps (int): The number of times the parameters have been
                updated.
        """
        momentum = (1 - self.momentum) * math.exp(
            -float(1 + steps) / self.gamma) + self.momentum
        averaged_param.assign_(averaged_param * (1 - momentum) + source_param * momentum)


# In[ ]:





# In[24]:


class LabelEncoder:
    def __init__(self) -> None:
        self.classes_: dict[Union[str, Real], int] = {}
        self.inverse_classes_: dict[int,Union [str, Real]] = {}
        self.num_classes: int = 0
        self.is_numerical: bool = True
    def update(self, y: Union[Sequence[Any], np.ndarray]):
        """Update the encoder with new labels encountered after initial fitting.

        This method identifies labels in the input sequence that are not already
        known to the encoder and adds them to the internal mapping. It does not
        change the encoding of previously seen labels.

        Args:
            y (Sequence[Any] | np.ndarray): A sequence or array of potentially new labels.

        Returns:
            LabelEncoder: The updated encoder instance.

        """
        if self.is_numerical:
            # Do not update if the original data was purely numerical
            return self

        # Standardize input type to list for easier processing
        if isinstance(y, np.ndarray):
            input_labels = y.flatten().tolist()
        elif isinstance(y, Sequence) and not isinstance(y, str):
            input_labels = list(y)
        elif y is None:
            # Handle cases where a label field might be None or empty
            return self
        else:
            # Handle single item case or string (treat string as single label)
            input_labels = [y]

        # Find labels not already in the encoder efficiently using sets
        current_labels_set = set(self.classes_.keys())
        new_unique_labels = set(input_labels) - current_labels_set

        if not new_unique_labels:
            # No new labels to add
            return self

        # Separate and sort new labels for deterministic encoding order
        numeric_labels: list[Real] = []
        string_labels: list[str] = []

        for label in new_unique_labels:
            (numeric_labels if isinstance(label, Real) else string_labels).append(label)
        sorted_new_labels = sorted(numeric_labels) + sorted(string_labels, key=str)

        for label in sorted_new_labels:
            new_id = self.num_classes
            self.classes_[label] = new_id
            self.inverse_classes_[new_id] = label
            self.num_classes += 1

        return self
    def inverse_transform(self, y: Union[Sequence[Any], np.ndarray]) -> np.ndarray:
        """Transform encoded indices back to original labels.

        Args:
            y (Sequence[Any] | np.ndarray): Encoded integer indices.

        Returns:
            np.ndarray: Original labels.

        """
        if isinstance(y, np.ndarray):
            y = y.flatten().tolist()

        if self.is_numerical:
            return np.array(y)

        return np.array([self.inverse_classes_[label] for label in y])
    def fit(self, y: Union[Sequence[Any], np.ndarray]):
        """Fit the encoder to the input labels.

        Args:
            y (Sequence[Any] | np.ndarray): Input labels to fit the encoder.

        Returns:
            LabelEncoder: The fitted encoder instance.

        """
        if isinstance(y, np.ndarray):
            y = y.flatten().tolist()

        # If input is empty, default to non-numerical to allow potential updates later
        if not y:
            self.is_numerical = False
            return self

        self.is_numerical = all(isinstance(label, Real) for label in y)

        if self.is_numerical:
            return self

        unique_labels = sorted(set(y), key=self.custom_sort)
        for label in unique_labels:
            if label not in self.classes_:
                self.classes_[label] = self.num_classes
                self.inverse_classes_[self.num_classes] = label
                self.num_classes += 1
        return self
    def transform(self, y: Union[Sequence[Any], np.ndarray]) -> np.ndarray:
        """Transform labels to encoded integer indices.

        Args:
            y (Sequence[Any] | np.ndarray): Input labels to transform.

        Returns:
            np.ndarray: Encoded integer indices.

        """
        if isinstance(y, np.ndarray):
            y = y.flatten().tolist()

        if self.is_numerical:
            return np.array(y)

        return np.array([self.classes_[label] for label in y])
    def custom_sort(item: Any) -> Tuple[int, Union[Real, str]]:
        """Sort items by type then value for consistent label ordering.

        This function is used to sort labels in a consistent order, prioritizing numerical
        values before string values. All numerical values are given priority 0, while
        string values are given priority 1, ensuring numerical values are sorted first.

        Args:
            item (Any): Item to be sorted, can be either a numeric value or any other type.

        Returns:
            tuple[int, Real | str]: A tuple with sort priority (0 for numbers, 1 for others)
                and the value itself (or string representation for non-numeric values).

        """
        return (0, item) if isinstance(item, Real) else (1, str(item))


# In[ ]:





# In[25]:


@dataclass
class LabelMetadata:
    """Stores metadata about a label field."""

    input_type: type
    is_numerical: bool
    dtype: Union[np.dtype, None] = None
    encoder: Union[LabelEncoder, None] = None


# In[ ]:





# In[26]:


class LabelManager:
    # Notice: Althrough use np.ndarray pass information, this class do not has other relies,so it almost the origin
    def __init__(self) -> None:
        self.metadata: dict[str, dict[str, LabelMetadata]] = defaultdict(dict)
    def process_field(self, data_name: str, label_field: str, field_data: Any) -> np.ndarray:
        """Process a label field, store metadata, and encode.

        If the field has been processed before (metadata exists), this will update
        the existing LabelEncoder with any new labels found in `field_data` before encoding.
        Otherwise, it analyzes the input, creates metadata, and fits the encoder.

        Args:
            data_name (str): The name of the main data type (e.g., 'bboxes', 'keypoints').
            label_field (str): The specific label field being processed (e.g., 'class_labels').
            field_data (Any): The actual label data for this field.

        Returns:
            np.ndarray: The encoded label data as a numpy array.

        """
        if data_name in self.metadata and label_field in self.metadata[data_name]:
            # Metadata exists, potentially update encoder
            metadata = self.metadata[data_name][label_field]
            if not metadata.is_numerical and metadata.encoder:
                metadata.encoder.update(field_data)
        else:
            # First time seeing this field, analyze and create metadata
            metadata = self._analyze_input(field_data)
            self.metadata[data_name][label_field] = metadata

        # Encode data using the (potentially updated) metadata/encoder
        return self._encode_data(field_data, metadata)

    def restore_field(self, data_name: str, label_field: str, encoded_data: np.ndarray) -> Any:
        """Restore a label field to its original format."""
        metadata = self.metadata[data_name][label_field]
        decoded_data = self._decode_data(encoded_data, metadata)
        return self._restore_type(decoded_data, metadata)
    def _decode_data(self, encoded_data: np.ndarray, metadata: LabelMetadata) -> np.ndarray:
        """Decode processed data."""
        if metadata.is_numerical:
            if metadata.dtype is not None:
                return encoded_data.astype(metadata.dtype)
            return encoded_data.flatten()  # Flatten for list conversion

        if metadata.encoder is None:
            raise ValueError("Encoder not found for non-numerical data")

        decoded = metadata.encoder.inverse_transform(encoded_data.astype(int))
        return decoded.reshape(-1)  # Ensure 1D array
    def _restore_type(self, decoded_data: np.ndarray, metadata: LabelMetadata) -> Any:
        """Restore data to its original type."""
        # If original input was a list or sequence, convert back to list
        if isinstance(metadata.input_type, type) and issubclass(metadata.input_type, (list, Sequence)):
            return decoded_data.tolist()

        # If original input was a numpy array, restore original dtype
        if isinstance(metadata.input_type, type) and issubclass(metadata.input_type, np.ndarray):
            if metadata.dtype is not None:
                return decoded_data.astype(metadata.dtype)
            return decoded_data

        # For any other type, convert to list by default
        return decoded_data.tolist()
    def _encode_data(self, field_data: Any, metadata: LabelMetadata) -> np.ndarray:
        """Encode field data for processing."""
        if metadata.is_numerical:
            # For numerical values, convert to float32 for processing
            if isinstance(field_data, np.ndarray):
                return field_data.reshape(-1, 1).astype(np.float32)
            return np.array(field_data, dtype=np.float32).reshape(-1, 1)

        # For non-numerical values, use LabelEncoder
        if metadata.encoder is None:
            raise ValueError("Encoder not initialized for non-numerical data")
        return metadata.encoder.fit_transform(field_data).reshape(-1, 1)
    def _analyze_input(self, field_data: Any) -> LabelMetadata:
        """Analyze input data and create metadata."""
        input_type = type(field_data)
        dtype = field_data.dtype if isinstance(field_data, np.ndarray) else None

        # Determine if input is numerical. Handle empty case explicitly.
        if isinstance(field_data, np.ndarray) and field_data.size > 0:
            is_numerical = np.issubdtype(field_data.dtype, np.number)
        elif isinstance(field_data, Sequence) and not isinstance(field_data, str) and field_data:
            is_numerical = all(isinstance(label, (int, float)) for label in field_data)
        elif isinstance(field_data, (int, float)):
            is_numerical = True  # Handle single numeric item
        else:
            # Default to non-numerical for empty sequences, single strings, or other types
            is_numerical = False

        metadata = LabelMetadata(
            input_type=input_type,
            is_numerical=is_numerical,
            dtype=dtype,
        )

        if not is_numerical:
            metadata.encoder = LabelEncoder()

        return metadata


# In[ ]:





# In[27]:


class BboxProcessor:
    ShapeType = Dict[Literal["depth", "height", "width"], int]
    def __init__(self, params: BboxParams, additional_targets: Union[Dict[str, str], None] = None):
        self.params = params
        self.data_fields = [self.default_data_name]
        self.is_sequence_input: dict[str, bool] = {}
        self.label_manager = LabelManager()
        assert additional_targets is None
    def default_data_name(self) -> str:
        """Returns the default key for bounding box data in transformations.

        Returns:
            str: The string 'bboxes'.

        """
        return "bboxes"
    def ensure_data_valid(self, data: Dict[str, Any]) -> None:
        """Validates the input bounding box data.

        Checks that:
        - Bounding boxes have labels (either in the bbox array or in label_fields)
        - All specified label_fields exist in the data

        Args:
            data (dict[str, Any]): Dict with bounding boxes and optional label fields.

        Raises:
            ValueError: If bounding boxes don't have labels or if label_fields are invalid.

        """
        if self.params.label_fields and not all(i in data for i in self.params.label_fields):
            msg = "Your 'label_fields' are not valid - them must have same names as params in dict"
            raise ValueError(msg)
    def preprocess(self, data: Dict[str, Any]) -> None:
        """Process data before transformation.

        Args:
            data (dict[str, Any]): Data dictionary to preprocess.

        """
        shape = get_shape(data)

        for data_name in set(self.data_fields) & set(data.keys()):  # Convert list of lists to numpy array if necessary
            if isinstance(data[data_name], Sequence):
                self.is_sequence_input[data_name] = True
                data[data_name] = np.array(data[data_name], dtype=np.float32)
            else:
                self.is_sequence_input[data_name] = False

        data = self.add_label_fields_to_data(data)
        for data_name in set(self.data_fields) & set(data.keys()):
            data[data_name] = self.check_and_convert(data[data_name], shape, direction="to")
    def add_label_fields_to_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Add label fields to data arrays.

        This method processes label fields and joins them with the corresponding data arrays.

        Args:
            data (dict[str, Any]): Input data dictionary.

        Returns:
            dict[str, Any]: Data with label fields added.

        """
        if not self.params.label_fields:
            return data

        for data_name in set(self.data_fields) & set(data.keys()):
            if not data[data_name].size:
                continue
            data_array = data[data_name]
            if self.params.label_fields is not None:
                for label_field in self.params.label_fields:
                    if len(data[data_name]) != len(data[label_field]):
                        raise ValueError(
                            f"The lengths of {data_name} and {label_field} do not match. "
                            f"Got {len(data[data_name])} and {len(data[label_field])} respectively.",
                                )
                    encoded_labels = self.label_manager.process_field(data_name, label_field, data[label_field])
                    data_array = np.hstack((data_array, encoded_labels))
                    del data[label_field]
                    data[data_name] = data_array
                return data
    def check_and_convert(
        self,
        data: np.ndarray,
        shape: ShapeType,
        direction: Literal["to", "from"] = "to",
    ) -> np.ndarray:
        """Converts bounding boxes between formats and applies preprocessing/postprocessing.

        Args:
            data (np.ndarray): Array of bounding boxes to process.
            shape (ShapeType): Image shape as dict with height and width keys.
            direction (Literal["to", "from"]): Direction of conversion:
                - "to": Convert from original format to albumentations format
                - "from": Convert from albumentations format to original format
                Default: "to".

        Returns:
            np.ndarray: Processed bounding boxes.

        Note:
            When direction="to":
            1. Converts to albumentations format
            2. Clips boxes if params.clip=True
            3. Filters invalid boxes if params.filter_invalid_bboxes=True
            4. Validates remaining boxes

            When direction="from":
            1. Validates boxes
            2. Converts back to original format

        """
        if direction == "to":
            # First convert to albumentations format
            if self.params.format == "albumentations":
                converted_data = data
            else:
                converted_data = convert_bboxes_to_albumentations(
                    data,
                    self.params.format,
                    shape,
                    check_validity=False,  # Don't check validity yet
                )

            if self.params.clip and converted_data.size > 0:
                converted_data[:, :4] = np.clip(converted_data[:, :4], 0, 1)

            # Then filter invalid boxes if requested
            if self.params.filter_invalid_bboxes:
                converted_data = filter_bboxes(
                    converted_data,
                    shape,
                    min_area=0,
                    min_visibility=0,
                    min_width=0,
                    min_height=0,
                )

            # Finally check the remaining boxes
            self.check_bboxes(converted_data, shape)
            return converted_data
        self.check_bboxes(data, shape)
        if self.params.format == "albumentations":
            return data
        return convert_bboxes_from_albumentations(data, self.params.format, shape)
    def check_bboxes(bboxes: np.ndarray, shape=None) -> None:
        
        """Check if bounding boxes are valid.

        Args:
            bboxes (np.ndarray): A numpy array of bounding boxes with shape (num_bboxes, 4+).

        Raises:
            ValueError: If any bounding box is invalid.

        """
        # Check if all values are in range [0, 1]
        in_range = (bboxes[:, :4] >= 0) & (bboxes[:, :4] <= 1)
        close_to_zero = np.isclose(bboxes[:, :4], 0)
        close_to_one = np.isclose(bboxes[:, :4], 1)
        valid_range = in_range | close_to_zero | close_to_one

        if not np.all(valid_range):
            invalid_idx = np.where(~np.all(valid_range, axis=1))[0][0]
            invalid_bbox = bboxes[invalid_idx]
            invalid_coord = ["x_min", "y_min", "x_max", "y_max"][np.where(~valid_range[invalid_idx])[0][0]]
            invalid_value = invalid_bbox[np.where(~valid_range[invalid_idx])[0][0]]
            raise ValueError(
                f"Expected {invalid_coord} for bbox {invalid_bbox} to be in the range [0.0, 1.0], got {invalid_value}.",
            )

        # Check if x_max > x_min and y_max > y_min
        valid_order = (bboxes[:, 2] > bboxes[:, 0]) & (bboxes[:, 3] > bboxes[:, 1])

        if not np.all(valid_order):
            invalid_idx = np.where(~valid_order)[0][0]
            invalid_bbox = bboxes[invalid_idx]
            if invalid_bbox[2] <= invalid_bbox[0]:
                raise ValueError(f"x_max is less than or equal to x_min for bbox {invalid_bbox}.")

            raise ValueError(f"y_max is less than or equal to y_min for bbox {invalid_bbox}.")
    def convert_from_albumentations(self, data: np.ndarray, shape: ShapeType) -> np.ndarray:
        """Convert bounding boxes from internal Albumentations format to the specified format.

        Args:
            data (np.ndarray): Bounding boxes in Albumentations format.
            shape (ShapeType): Shape information for validation.

        Returns:
            np.ndarray: Converted bounding boxes in the target format.

        """
        return np.array(
            convert_bboxes_from_albumentations(data, self.params.format, shape, check_validity=True),
            dtype=data.dtype,
        )

    def convert_to_albumentations(self, data: np.ndarray, shape: ShapeType) -> np.ndarray:
        """Convert bounding boxes from the specified format to internal Albumentations format.

        Args:
            data (np.ndarray): Bounding boxes in source format.
            shape (ShapeType): Shape information for validation.

        Returns:
            np.ndarray: Converted bounding boxes in Albumentations format.

        """
        if self.params.clip:
            data_np = convert_bboxes_to_albumentations(data, self.params.format, shape, check_validity=False)
            data_np = filter_bboxes(data_np, shape, min_area=0, min_visibility=0, min_width=0, min_height=0)
            check_bboxes(data_np)
            return data_np

        return convert_bboxes_to_albumentations(data, self.params.format, shape, check_validity=True)
    def filter_bboxes(
    bboxes: np.ndarray,
    shape: ShapeType,
    min_area: float = 0.0,
    min_visibility: float = 0.0,
    min_width: float = 1.0,
    min_height: float = 1.0,
    max_accept_ratio: Union[float, None] = None,
    ) -> np.ndarray:
        
        """Remove bounding boxes that either lie outside of the visible area by more than min_visibility
        or whose area in pixels is under the threshold set by `min_area`. Also crops boxes to final image size.

        Args:
            bboxes (np.ndarray): A numpy array of bounding boxes with shape (num_bboxes, 4+).
            shape (ShapeType): The shape of the image/volume:
                               - For 2D: {'height': int, 'width': int}
                               - For 3D: {'height': int, 'width': int, 'depth': int}
            min_area (float): Minimum area of a bounding box in pixels. Default: 0.0.
            min_visibility (float): Minimum fraction of area for a bounding box to remain. Default: 0.0.
            min_width (float): Minimum width of a bounding box in pixels. Default: 0.0.
            min_height (float): Minimum height of a bounding box in pixels. Default: 0.0.
            max_accept_ratio (float | None): Maximum allowed aspect ratio, calculated as max(width/height, height/width).
                Boxes with higher ratios will be filtered out. Default: None.

        Returns:
            np.ndarray: Filtered bounding boxes.

        """
        epsilon = 1e-7

        if len(bboxes) == 0:
            return np.array([], dtype=np.float32).reshape(0, 4)

        # Calculate areas of bounding boxes before clipping in pixels
        denormalized_box_areas = calculate_bbox_areas_in_pixels(bboxes, shape)

        # Clip bounding boxes in ratio
        clipped_bboxes = clip_bboxes(bboxes, shape)

        # Calculate areas of clipped bounding boxes in pixels
        clipped_box_areas = calculate_bbox_areas_in_pixels(clipped_bboxes, shape)

        # Calculate width and height of the clipped bounding boxes
        denormalized_bboxes = denormalize_bboxes(clipped_bboxes[:, :4], shape)

        clipped_widths = denormalized_bboxes[:, 2] - denormalized_bboxes[:, 0]
        clipped_heights = denormalized_bboxes[:, 3] - denormalized_bboxes[:, 1]

        # Calculate aspect ratios if needed
        if max_accept_ratio is not None:
            aspect_ratios = np.maximum(
                clipped_widths / (clipped_heights + epsilon),
                clipped_heights / (clipped_widths + epsilon),
            )
            valid_ratios = aspect_ratios <= max_accept_ratio
        else:
            valid_ratios = np.ones_like(denormalized_box_areas, dtype=bool)

        # Create a mask for bboxes that meet all criteria
        mask = (
            (denormalized_box_areas >= epsilon)
            & (clipped_box_areas >= min_area - epsilon)
            & (clipped_box_areas / (denormalized_box_areas + epsilon) >= min_visibility)
            & (clipped_widths >= min_width - epsilon)
            & (clipped_heights >= min_height - epsilon)
            & valid_ratios
        )

        # Apply the mask to get the filtered bboxes
        filtered_bboxes = clipped_bboxes[mask]

        return np.array([], dtype=np.float32).reshape(0, 4) if len(filtered_bboxes) == 0 else filtered_bboxes
    def filter(self, data: np.ndarray, shape: ShapeType) -> np.ndarray:
        """Filter bounding boxes based on size and visibility criteria.

        Args:
            data (np.ndarray): Array of bounding boxes in Albumentations format.
            shape (ShapeType): Shape information for validation.

        Returns:
            np.ndarray: Filtered bounding boxes that meet the criteria.

        """
        self.params: BboxParams
        return filter_bboxes(
            data,
            shape,
            min_area=self.params.min_area,
            min_visibility=self.params.min_visibility,
            min_width=self.params.min_width,
            min_height=self.params.min_height,
            max_accept_ratio=self.params.max_accept_ratio,
        )
    def normalize_bboxes(bboxes: np.ndarray, shape: Union[ShapeType, Tuple[int, int]]) -> np.ndarray:
        """Normalize array of bounding boxes.

        Args:
            bboxes (np.ndarray): Denormalized bounding boxes `[(x_min, y_min, x_max, y_max, ...)]`.
            shape (ShapeType | tuple[int, int]): Image shape `(height, width)`.

        Returns:
            np.ndarray: Normalized bounding boxes `[(x_min, y_min, x_max, y_max, ...)]`.

        """
        if isinstance(shape, tuple):
            rows, cols = shape[:2]
        else:
            rows, cols = shape["height"], shape["width"]

        normalized = bboxes.copy().astype(float)
        normalized[:, [0, 2]] /= cols
        normalized[:, [1, 3]] /= rows
        return normalized
    def denormalize_bboxes(
    bboxes: np.ndarray,
    shape:Union[ShapeType, Tuple[int, int]],
    ) -> np.ndarray:
        """Denormalize array of bounding boxes.

        Args:
            bboxes (np.ndarray): Normalized bounding boxes `[(x_min, y_min, x_max, y_max, ...)]`.
            shape (ShapeType | tuple[int, int]): Image shape `(height, width)`.

        Returns:
            np.ndarray: Denormalized bounding boxes `[(x_min, y_min, x_max, y_max, ...)]`.

        """
        scale_factors = (shape[1], shape[0]) if isinstance(shape, tuple) else (shape["width"], shape["height"])

        # Vectorized scaling of bbox coordinates
        return bboxes * np.array([*scale_factors, *scale_factors, *[1] * (bboxes.shape[1] - 4)], dtype=float)
    def convert_bboxes_to_albumentations(
        bboxes: np.ndarray,
        source_format: Literal["coco", "pascal_voc", "yolo"],
        shape: ShapeType,
        check_validity: bool = False,
    ) -> np.ndarray:
        """Convert bounding boxes from a specified format to the format used by albumentations:
        normalized coordinates of top-left and bottom-right corners of the bounding box in the form of
        `(x_min, y_min, x_max, y_max)` e.g. `(0.15, 0.27, 0.67, 0.5)`.

        Args:
            bboxes (np.ndarray): A numpy array of bounding boxes with shape (num_bboxes, 4+).
            source_format (Literal["coco", "pascal_voc", "YOLO"]): Format of the input bounding boxes.
            shape (ShapeType): Image shape (height, width).
            check_validity (bool): Check if all boxes are valid boxes.

        Returns:
            np.ndarray: An array of bounding boxes in albumentations format with shape (num_bboxes, 4+).

        Raises:
            ValueError: If `source_format` is not 'coco', 'pascal_voc', or 'YOLO'.
            ValueError: If in YOLO format, any coordinates are not in the range (0, 1].

        """
        if source_format not in {"coco", "pascal_voc", "yolo"}:
            raise ValueError(
                f"Unknown source_format {source_format}. Supported formats are: 'coco', 'pascal_voc' and 'yolo'",
            )

        bboxes = bboxes.copy().astype(np.float32)
        converted_bboxes = np.zeros_like(bboxes)
        converted_bboxes[:, 4:] = bboxes[:, 4:]  # Preserve additional columns

        if source_format == "coco":
            converted_bboxes[:, 0] = bboxes[:, 0]  # x_min
            converted_bboxes[:, 1] = bboxes[:, 1]  # y_min
            converted_bboxes[:, 2] = bboxes[:, 0] + bboxes[:, 2]  # x_max
            converted_bboxes[:, 3] = bboxes[:, 1] + bboxes[:, 3]  # y_max
        elif source_format == "yolo":
            if check_validity and np.any((bboxes[:, :4] <= 0) | (bboxes[:, :4] > 1)):
                raise ValueError(f"In yolo format all coordinates must be float and in range (0, 1], got {bboxes}")

            w_half, h_half = bboxes[:, 2] / 2, bboxes[:, 3] / 2
            converted_bboxes[:, 0] = bboxes[:, 0] - w_half  # x_min
            converted_bboxes[:, 1] = bboxes[:, 1] - h_half  # y_min
            converted_bboxes[:, 2] = bboxes[:, 0] + w_half  # x_max
            converted_bboxes[:, 3] = bboxes[:, 1] + h_half  # y_max
        else:  # pascal_voc
            converted_bboxes[:, :4] = bboxes[:, :4]

        if source_format != "yolo":
            converted_bboxes[:, :4] = normalize_bboxes(converted_bboxes[:, :4], shape)

        if check_validity:
            check_bboxes(converted_bboxes)

        return converted_bboxes
    def convert_bboxes_from_albumentations(
    bboxes: np.ndarray,
    target_format: Literal["coco", "pascal_voc", "yolo"],
    shape: ShapeType,
    check_validity: bool = False,
    ) -> np.ndarray:
        """Convert bounding boxes from the format used by albumentations to a specified format.

        Args:
            bboxes (np.ndarray): A numpy array of albumentations bounding boxes with shape (num_bboxes, 4+).
                    The first 4 columns are [x_min, y_min, x_max, y_max].
            target_format (Literal["coco", "pascal_voc", "YOLO"]): Required format of the output bounding boxes.
            shape (ShapeType): Image shape (height, width).
            check_validity (bool): Check if all boxes are valid boxes.

        Returns:
            np.ndarray: An array of bounding boxes in the target format with shape (num_bboxes, 4+).

        Raises:
            ValueError: If `target_format` is not 'coco', 'pascal_voc' or 'YOLO'.

        """
        if target_format not in {"coco", "pascal_voc", "yolo"}:
            raise ValueError(
                f"Unknown target_format {target_format}. Supported formats are: 'coco', 'pascal_voc' and 'yolo'",
            )

        if check_validity:
            check_bboxes(bboxes)

        converted_bboxes = np.zeros_like(bboxes)
        converted_bboxes[:, 4:] = bboxes[:, 4:]  # Preserve additional columns

        denormalized_bboxes = denormalize_bboxes(bboxes[:, :4], shape) if target_format != "yolo" else bboxes[:, :4]

        if target_format == "coco":
            converted_bboxes[:, 0] = denormalized_bboxes[:, 0]  # x_min
            converted_bboxes[:, 1] = denormalized_bboxes[:, 1]  # y_min
            converted_bboxes[:, 2] = denormalized_bboxes[:, 2] - denormalized_bboxes[:, 0]  # width
            converted_bboxes[:, 3] = denormalized_bboxes[:, 3] - denormalized_bboxes[:, 1]  # height
        elif target_format == "yolo":
            converted_bboxes[:, 0] = (denormalized_bboxes[:, 0] + denormalized_bboxes[:, 2]) / 2  # x_center
            converted_bboxes[:, 1] = (denormalized_bboxes[:, 1] + denormalized_bboxes[:, 3]) / 2  # y_center
            converted_bboxes[:, 2] = denormalized_bboxes[:, 2] - denormalized_bboxes[:, 0]  # width
            converted_bboxes[:, 3] = denormalized_bboxes[:, 3] - denormalized_bboxes[:, 1]  # height
        else:  # pascal_voc
            converted_bboxes[:, :4] = denormalized_bboxes

        return converted_bboxes
    def clip_bboxes(bboxes: np.ndarray, shape: ShapeType) -> np.ndarray:
        """Clip bounding boxes to the image shape.

        Args:
            bboxes (np.ndarray): A numpy array of bounding boxes with shape (num_bboxes, 4+).
            shape (ShapeType): The shape of the image/volume:
                               - For 2D: {'height': int, 'width': int}
                               - For 3D: {'height': int, 'width': int, 'depth': int}

        Returns:
            np.ndarray: A numpy array of bounding boxes with shape (num_bboxes, 4+).

        """
        height, width = shape["height"], shape["width"]

            # Denormalize bboxes
        denorm_bboxes = denormalize_bboxes(bboxes, shape)

    ## Note:
    # It could be tempting to use cols - 1 and rows - 1 as the upper bounds for the clipping

    # But this would cause the bounding box to be clipped to the image dimensions - 1 which is not what we want.
    # Bounding box lives not in the middle of pixels but between them.

    # Examples: for image with height 100, width 100, the pixel values are in the range [0, 99]
    # but if we want bounding box to be 1 pixel width and height and lie on the boundary of the image
    # it will be described as [99, 99, 100, 100] => clip by image_size - 1 will lead to [99, 99, 99, 99]
    # which is incorrect

    # It could be also tempting to clip `x_min`` to `cols - 1`` and `y_min` to `rows - 1`, but this also leads
    # to another error. If image fully lies outside of the visible area and min_area is set to 0, then
    # the bounding box will be clipped to the image size - 1 and will be 1 pixel in size and fully visible,
    # but it should be completely removed.

    # Clip coordinates
        denorm_bboxes[:, [0, 2]] = np.clip(denorm_bboxes[:, [0, 2]], 0, width, out=denorm_bboxes[:, [0, 2]])
        denorm_bboxes[:, [1, 3]] = np.clip(denorm_bboxes[:, [1, 3]], 0, height, out=denorm_bboxes[:, [1, 3]])

            # Normalize clipped bboxes
        return normalize_bboxes(denorm_bboxes, shape)
    def postprocess(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process data after transformation.

        Args:
            data (dict[str, Any]): Data dictionary after transformation.

        Returns:
            dict[str, Any]: Processed data dictionary.

        """
        shape = get_shape(data)
        data = self._process_data_fields(data, shape)
        data = self.remove_label_fields_from_data(data)
        return self._convert_sequence_inputs(data)
    def _process_data_fields(self, data: Dict[str, Any], shape: ShapeType) -> Dict[str, Any]:
        for data_name in set(self.data_fields) & set(data.keys()):
            field_data = self.filter(data[data_name], shape)
            data[data_name] = check_and_convert(field_data, shape, direction="from")
        return data
    def remove_label_fields_from_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Remove label fields from data arrays and restore them as separate entries.

            Args:
                data (dict[str, Any]): Input data dictionary with combined label fields.

            Returns:
                dict[str, Any]: Data with label fields extracted as separate entries.

        """
        if not self.params.label_fields:
            return data

        for data_name in set(self.data_fields) & set(data.keys()):
            if not data[data_name].size:
                if self.params.label_fields is not None:
                    for label_field in self.params.label_fields:
                        data[label_field] = []
                continue
            if self.params.label_fields is None:
                continue


            data_array = data[data_name]
            num_label_fields = len(self.params.label_fields)
            non_label_columns = data_array.shape[1] - num_label_fields

            for idx, label_field in enumerate(self.params.label_fields):
                encoded_labels = data_array[:, non_label_columns + idx]
                data[label_field] = self.label_manager.restore_field(data_name, label_field, encoded_labels)

            data[data_name] = data_array[:, :non_label_columns]

        return data


# In[ ]:





# In[28]:


class Compose:
    AVAILABLE_KEYS = ("image", "mask", "masks", "bboxes", "keypoints", "volume", "volumes", "mask3d", "masks3d")
    MASK_KEYS = (
    "mask",  # 2D mask
    "masks",  # Multiple 2D masks
    "mask3d",  # 3D mask
    "masks3d",  # Multiple 3D masks
    )

    # Keys related to image data
    IMAGE_KEYS = {"image", "images"}
    CHECKED_SINGLE = {"image", "mask"}
    CHECKED_MULTI = {"masks", "images", "volumes", "masks3d"}
    CHECK_BBOX_PARAM = {"bboxes"}
    CHECK_KEYPOINTS_PARAM = {"keypoints"}
    VOLUME_KEYS = {"volume", "volumes"}
    CHECKED_VOLUME = {"volume"}
    CHECKED_VOLUMES = {"volumes"}
    CHECKED_MASK3D = {"mask3d"}
    CHECKED_MASKS3D = {"masks3d"}
    def __init__(
        self,
        transforms,
        bbox_params: Union[Dict[str, Any],BboxParams,None] = None,
        keypoint_params: Union[Dict[str, Any],Any,None] = None,
        additional_targets: Union[Dict[str, str],None] = None,
        p: float = 1.0,
        is_check_shapes: bool = True,
        strict: bool = False,
        mask_interpolation: Union[int, None] = None,
        seed: Union[int, None] = None,
        save_applied_params: bool = False,
    ):
         # Store the original base seed for worker context recalculation
        self._base_seed = seed

        # Get effective seed considering worker context
#         effective_seed = self._get_effective_seed(seed)

        self.transforms = transforms
        self.p = p

        self.replay_mode = False
        self._additional_targets: Dict[str, str] = {}
        self._available_keys: set[str] = set()
        self.processors: Dict[str, BboxProcessor] = {}
#         self._set_keys()
#         self.set_mask_interpolation(mask_interpolation)
#         self.set_random_seed(seed)
        self.save_applied_params = save_applied_params

        if bbox_params:
            if isinstance(bbox_params, dict):
                b_params = BboxParams(**bbox_params)
            elif isinstance(bbox_params, BboxParams):
                b_params = bbox_params
            else:
                msg = "unknown format of bbox_params, please use `Dict` or `BboxParams`"
                raise ValueError(msg)
            self.processors["bboxes"] = BboxProcessor(b_params)

        if keypoint_params:
            raise NotImplementedError
#             if isinstance(keypoint_params, Dict):
#                 k_params = KeypointParams(**keypoint_params)
#             elif isinstance(keypoint_params, KeypointParams):
#                 k_params = keypoint_params
#             else:
#                 msg = "unknown format of keypoint_params, please use `Dict` or `KeypointParams`"
#                 raise ValueError(msg)
#             self.processors["keypoints"] = KeypointsProcessor(k_params)

#         for proc in self.processors.values():
#             proc.ensure_transforms_valid(self.transforms)

#         self.add_targets(additional_targets)
        if not self.transforms:  # if no transforms -> do nothing, all keys will be available
            self._available_keys.update(AVAILABLE_KEYS)
        self.main_compose = True
        self.is_check_args = True
        self.strict = strict

        self.is_check_shapes = is_check_shapes
        self.check_each_transform = tuple(  # processors that checks after each transform
            proc for proc in self.processors.values() if getattr(proc.params, "check_each_transform", False)
        )
#         self._set_check_args_for_transforms(self.transforms)

#         self._set_processors_for_transforms(self.transforms)

        self.save_applied_params = save_applied_params
        self._images_was_list = False
        self._masks_was_list = False
        self._last_torch_seed: Union[int, None] = None
    def __call__(self, *args: Any, force_apply: bool = False, **data: Any) -> Dict[str, Any]:
        """Apply transformations to data with automatic worker seed synchronization.

        Args:
            *args (Any): Positional arguments are not supported.
            force_apply (bool): Whether to apply transforms regardless of probability. Default: False.
            **data (Any): Dict with data to transform.

        Returns:
            Dict[str, Any]: Dictionary with transformed data.

        Raises:
            KeyError: If positional arguments are provided.

        """
        # Check and sync worker seed if needed
#         self._check_worker_seed()

        if args:
            msg = "You have to pass data to augmentations as named arguments, for example: aug(image=image)"
            raise KeyError(msg)

        # Initialize applied_transforms only in top-level Compose if requested
        if self.save_applied_params and self.main_compose:
            data["applied_transforms"] = []
        self.py_random = random.random()
        need_to_run = force_apply or random.random() < self.p
        if not need_to_run:
            return data

        self.preprocess(data)

        for t in self.transforms:
            data = t(**data)
            self._track_transform_params(t, data)
            data = self.check_data_post_transform(data)

        return self.postprocess(data)
    def preprocess(self, data: Any) -> None:
        """Preprocess input data before applying transforms."""
        # Always validate shapes if is_check_shapes is True, regardless of strict mode
        if self.is_check_shapes:
            shapes = []  # For H,W checks
            volume_shapes = []  # For D,H,W checks

            for data_name, data_value in data.items():
                internal_name = self._additional_targets.get(data_name, data_name)

                # Skip empty data
                if data_value is None:
                    continue

                shape = self._get_data_shape(data_name, internal_name, data_value)
                if shape is not None:
                    if internal_name in Union[CHECKED_VOLUME, CHECKED_MASK3D]:
                        shapes.append(shape[1:3])  # H,W from (D,H,W)
                        volume_shapes.append(shape[:3])  # D,H,W
                    elif internal_name in {"volumes", "masks3d"}:
                        shapes.append(shape[2:4])  # H,W from (N,D,H,W)
                        volume_shapes.append(shape[1:4])  # D,H,W from (N,D,H,W)
                    else:
                        shapes.append(shape[:2])  # H,W

            self._check_shape_consistency(shapes, volume_shapes)

        # Do strict validation only if enabled
        if self.strict:
            raise NotImplementedError
#             self._validate_data(data)

        self._preprocess_processors(data)
        self._preprocess_arrays(data)
    def _get_data_shape(self, data_name: str, internal_name: str, data: Any) -> Union[Tuple[int, ...], None]:
        """Get shape of data based on its type."""
        # Handle single images and masks
        if internal_name in CHECKED_SINGLE:
            if not isinstance(data, np.ndarray):
                raise TypeError(f"{data_name} must be numpy array type")
            return data.shape

        return None
    def _check_shape_consistency(self, shapes: List[Tuple[int, ...]], volume_shapes: List[Tuple[int, ...]]) -> None:
        """Check consistency of shapes."""
        # Check H,W consistency
        if self.is_check_shapes and shapes and shapes.count(shapes[0]) != len(shapes):
            raise ValueError(
                "Height and Width of image, mask or masks should be equal. You can disable shapes check "
                "by setting a parameter is_check_shapes=False of Compose class (do it only if you are sure "
                "about your data consistency).",
            )

        # Check D,H,W consistency for volumes and 3D masks
        if self.is_check_shapes and volume_shapes and volume_shapes.count(volume_shapes[0]) != len(volume_shapes):
            raise ValueError(
                "Depth, Height and Width of volume, mask3d, volumes and masks3d should be equal. "
                "You can disable shapes check by setting is_check_shapes=False.",
            )
    def _preprocess_processors(self, data: Dict[str, Any]) -> None:
        """Run preprocessors if this is the main compose."""
        if not self.main_compose:
            return
        for processor in self.processors.values():
            processor.ensure_data_valid(data)
        for processor in self.processors.values():
            processor.preprocess(data)
    def _preprocess_arrays(self, data: Dict[str, Any]) -> None:
        """Convert image lists to numpy arrays."""
        assert "mask" not in data
        if "images" not in data:
            return

        if isinstance(data["images"], (list, Tuple)):
            self._images_was_list = True
            # Skip stacking for empty lists
            if not data["images"]:
                return
            data["images"] = np.stack(data["images"])
        else:
            self._images_was_list = False
    def postprocess(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply post-processing to data after all transforms have been applied.

        Args:
            data (Dict[str, Any]): Data after transformation.

        Returns:
            Dict[str, Any]: Post-processed data.

        """
#         if self.main_compose:
#             for p in self.processors.values():
#                 p.postprocess(data)

#             # Convert back to list if original input was a list
#             if "images" in data and self._images_was_list:
#                 data["images"] = list(data["images"])

#             if "masks" in data and self._masks_was_list:
#                 data["masks"] = list(data["masks"])

        return data
    def _track_transform_params(self, transform, data: Dict[str, Any]) -> None:
        """Track transform parameters if tracking is enabled."""
        if "applied_transforms" in data and hasattr(transform, "params") and transform.params:
            data["applied_transforms"].append((transform.__class__.__name__, transform.params.copy()))
    def check_data_post_transform(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Check and filter data after transformation.

        Args:
            data (Dict[str, Any]): Dictionary containing transformed data

        Returns:
            Dict[str, Any]: Filtered data Dictionary

        """
        if self.check_each_transform:
            shape = get_shape(data)
            for proc in self.check_each_transform:
                for data_name, data_value in data.items():
                    if data_name in proc.data_fields or (
                        data_name in self._additional_targets
                        and self._additional_targets[data_name] in proc.data_fields
                    ):
                        data[data_name] = proc.filter(data_value, shape)
        return data


# In[29]:


#support albumentations class
albumentations = {'blur':Blur,
                  'medianblur':MedianBlur,
                  'clahe':CLAHE,
                  'bboxparams':BboxParams,
                  'togray':ToGray,
                  'bboxprocessor':BboxProcessor,
                  'labelmanager':LabelManager,
                  'labelmetadata':LabelMetadata,
                  'labelencoder':LabelEncoder,
                  'compose':Compose
} 


# In[30]:


class YOLOv5CocoDataset(Dataset):
    METAINFO = {
        'classes':
        ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
         'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
         'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
         'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
         'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
         'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
         'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
         'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
         'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
         'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
         'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
         'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
         'scissors', 'teddy bear', 'hair drier', 'toothbrush'),
        # palette is a list of color tuples, which is used for visualization.
        'palette':
        [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228),
         (0, 60, 100), (0, 80, 100), (0, 0, 70), (0, 0, 192), (250, 170, 30),
         (100, 170, 30), (220, 220, 0), (175, 116, 175), (250, 0, 30),
         (165, 42, 42), (255, 77, 255), (0, 226, 252), (182, 182, 255),
         (0, 82, 0), (120, 166, 157), (110, 76, 0), (174, 57, 255),
         (199, 100, 0), (72, 0, 118), (255, 179, 240), (0, 125, 92),
         (209, 0, 151), (188, 208, 182), (0, 220, 176), (255, 99, 164),
         (92, 0, 73), (133, 129, 255), (78, 180, 255), (0, 228, 0),
         (174, 255, 243), (45, 89, 255), (134, 134, 103), (145, 148, 174),
         (255, 208, 186), (197, 226, 255), (171, 134, 1), (109, 63, 54),
         (207, 138, 255), (151, 0, 95), (9, 80, 61), (84, 105, 51),
         (74, 65, 105), (166, 196, 102), (208, 195, 210), (255, 109, 65),
         (0, 143, 149), (179, 0, 194), (209, 99, 106), (5, 121, 0),
         (227, 255, 205), (147, 186, 208), (153, 69, 1), (3, 95, 161),
         (163, 255, 0), (119, 0, 170), (0, 182, 199), (0, 165, 120),
         (183, 130, 88), (95, 32, 0), (130, 114, 135), (110, 129, 133),
         (166, 74, 118), (219, 142, 185), (79, 210, 114), (178, 90, 62),
         (65, 70, 15), (127, 167, 115), (59, 105, 106), (142, 108, 45),
         (196, 172, 0), (95, 54, 80), (128, 76, 255), (201, 57, 1),
         (246, 0, 122), (191, 162, 208)]
        #  here comes some changes todo
        # 'classes': ("none", "pedestrian", "people", "bicycle", "car", "van", "truck", "tricycle", "awning-tricycle",
        #             "bus", "motor", "other"),
        # 'palette': [
        #     (100, 170, 30), (220, 220, 0), (220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228),
        #     (0, 60, 100), (0, 80, 100), (0, 0, 70), (0, 0, 192), (250, 170, 30)
        # ]
    }
    COCOAPI = COCO
    # ann_id is unique in coco dataset.
    ANN_ID_UNIQUE = True
    def __init__(self,
                 ann_file: Optional[str] = '',
                 metainfo: Union[Mapping, Any, None] = None,
                 seg_map_suffix: str = '.png',
                 proposal_file: Optional[str] = None,
                 file_client_args: Dict = None,
                 backend_args: Dict = None,
                 return_classes: bool = False,
                 caption_prompt: Optional[dict] = None,
                 data_root: Optional[str] = '',
                 data_prefix: Dict = dict(img_path=''),
                 batch_shapes_cfg: Optional[dict] = None,
                 filter_cfg: Optional[dict] = None,
                 indices: Optional[Union[int, Sequence[int]]] = None,
                 serialize_data: bool = True,
                 pipeline: List[Union[dict, Callable]] = [],
                 test_mode: bool = False,
                 lazy_init: bool = False,
                 max_refetch: int = 1000):
        msg = 'due to my energy is limited some function not supported'
        assert batch_shapes_cfg is None, msg
        self.batch_shapes_cfg = batch_shapes_cfg
        self.seg_map_suffix = seg_map_suffix
        self.proposal_file = proposal_file
        self.backend_args = backend_args
        self.return_classes = return_classes
        self.caption_prompt = caption_prompt
        if self.caption_prompt is not None:
            assert self.return_classes,                 'return_classes must be True when using caption_prompt'
        if file_client_args is not None:
            raise RuntimeError(
                'The `file_client_args` is deprecated, '
                'please use `backend_args` instead, please refer to'
                'https://github.com/open-mmlab/mmdetection/blob/main/configs/_base_/datasets/coco_detection.py'  # noqa: E501
            )
        self.ann_file = ann_file
        self._metainfo = self._load_metainfo(copy.deepcopy(metainfo))
        self.data_root = data_root
        self.data_prefix = copy.copy(data_prefix)
        self.filter_cfg = copy.deepcopy(filter_cfg)
        assert indices is None,msg
        self._indices = indices
        self.serialize_data = serialize_data
        self.test_mode = test_mode
        self.max_refetch = max_refetch
        self.data_list: List[dict] = []
        self.data_bytes: np.ndarray

        # Join paths.
        self._join_prefix()

        # Build pipeline.
        self.pipeline = Compose(pipeline)
        self._fully_initialized = False
        # Full initialize the dataset.
        if not lazy_init:
            self.full_init()
    @property
    def metainfo(self) -> dict:
        """Get meta information of dataset.

        Returns:
            dict: meta information collected from ``BaseDataset.METAINFO``,
            annotation file and metainfo argument during instantiation.
        """
        return copy.deepcopy(self._metainfo)
    def _load_metainfo(cls,
                       metainfo: Union[Mapping, Any, None] = None) -> dict:
        """Collect meta information from the dictionary of meta.

        Args:
            metainfo (Mapping or Config, optional): Meta information dict.
                If ``metainfo`` contains existed filename, it will be
                parsed by ``list_from_file``.

        Returns:
            dict: Parsed meta information.
        """
        # avoid `cls.METAINFO` being overwritten by `metainfo`
        cls_metainfo = copy.deepcopy(cls.METAINFO)
        if metainfo is None:
            return cls_metainfo
        if not isinstance(metainfo, Mapping):
            raise TypeError('metainfo should be a Mapping, '
                            f'but got {type(metainfo)}')

        for k, v in metainfo.items():
            if isinstance(v, str):
                # If type of value is string, and can be loaded from
                # corresponding backend. it means the file name of meta file.
                try:
                    cls_metainfo[k] = list_from_file(v)#todo
                except (TypeError, FileNotFoundError):
                    warning(
                        f'{v} is not a meta file, simply parsed as meta '
                        'information'
                    )
                    cls_metainfo[k] = v
            else:
                cls_metainfo[k] = v
        return cls_metainfo
    def _join_prefix(self):
        """Join ``self.data_root`` with ``self.data_prefix`` and
        ``self.ann_file``.

        Examples:
            >>> # self.data_prefix contains relative paths
            >>> self.data_root = 'a/b/c'
            >>> self.data_prefix = dict(img='d/e/')
            >>> self.ann_file = 'f'
            >>> self._join_prefix()
            >>> self.data_prefix
            dict(img='a/b/c/d/e')
            >>> self.ann_file
            'a/b/c/f'
            >>> # self.data_prefix contains absolute paths
            >>> self.data_root = 'a/b/c'
            >>> self.data_prefix = dict(img='/d/e/')
            >>> self.ann_file = 'f'
            >>> self._join_prefix()
            >>> self.data_prefix
            dict(img='/d/e')
            >>> self.ann_file
            'a/b/c/f'
        """
        # Automatically join annotation file path with `self.root` if
        # `self.ann_file` is not an absolute path.
        
        if self.ann_file and not is_abs(self.ann_file) and self.data_root:
#             self.ann_file = join_path(self.data_root, self.ann_file)
            self.ann_file = osp.join(self.data_root, self.ann_file)
        # Automatically join data directory with `self.root` if path value in
        # `self.data_prefix` is not an absolute path.
        for data_key, prefix in self.data_prefix.items():
            if not isinstance(prefix, str):
                raise TypeError('prefix should be a string, but got '
                                f'{type(prefix)}')
            if not is_abs(prefix) and self.data_root:
#                 self.data_prefix[data_key] = join_path(self.data_root, prefix)
                self.data_prefix[data_key] = osp.join(self.data_root, prefix)
            else:
                self.data_prefix[data_key] = prefix
    def full_init(self):
        """rewrite full_init() to be compatible with serialize_data in
        BatchShapePolicy."""
        if self._fully_initialized:
            return
        # load data information
        self.data_list = self.load_data_list()

        # batch_shapes_cfg
#         if self.batch_shapes_cfg:
#             batch_shapes_policy = TASK_UTILS.build(self.batch_shapes_cfg)
#             self.data_list = batch_shapes_policy(self.data_list)
#             del batch_shapes_policy

        # filter illegal data, such as data that has no annotations.
        self.data_list = self.filter_data()
        # Get subset data according to indices.
#         if self._indices is not None:
#             self.data_list = self._get_unserialized_subset(self._indices)

        # serialize data_list
        if self.serialize_data:
            self.data_bytes, self.data_address = self._serialize_data()

        self._fully_initialized = True
    def prepare_data(self, idx: int) -> Any:
        """Pass the dataset to the pipeline during training to support mixed
        data augmentation, such as Mosaic and MixUp."""
        if not self._fully_initialized:
            self.full_init()
        if self.test_mode is False:
            data_info = self.get_data_info(idx)
            data_info['dataset'] = self
            return self.pipeline(data_info)
        else:
            data_info = self.get_data_info(idx)
            return self.pipeline(data_info)
#             return super().prepare_data(idx)
    def __len__(self) -> int:
        """Get the length of filtered dataset and automatically call
        ``full_init`` if the  dataset has not been fully init.

        Returns:
            int: The length of filtered dataset.
        """
        if not self._fully_initialized:
            self.full_init()
        if self.serialize_data:
            return len(self.data_address)
        else:
            return len(self.data_list)
    def get_data_info(self, idx: int) -> dict:
        """Get annotation by index and automatically call ``full_init`` if the
        dataset has not been fully initialized.

        Args:
            idx (int): The index of data.

        Returns:
            dict: The idx-th annotation of the dataset.
        """
        if not self._fully_initialized:
            self.full_init()
        if self.serialize_data:
            start_addr = 0 if idx == 0 else self.data_address[idx - 1].item()
            end_addr = self.data_address[idx].item()
            bytes = memoryview(
                self.data_bytes[start_addr:end_addr])  # type: ignore
            data_info = pickle.loads(bytes)  # type: ignore
        else:
            data_info = copy.deepcopy(self.data_list[idx])
        # Some codebase needs `sample_idx` of data information. Here we convert
        # the idx to a positive number and save it in data information.
        if idx >= 0:
            data_info['sample_idx'] = idx
        else:
            data_info['sample_idx'] = len(self) + idx

        return data_info
    def load_data_list(self) -> List[dict]:
        """Load annotations from an annotation file named as ``self.ann_file``

        Returns:
            List[dict]: A list of annotation.
        """  # noqa: E501
#         print(type(self.ann_file))
        with get_local_path(self.ann_file) as local_path:
            self.coco = self.COCOAPI(local_path)
        # The order of returned `cat_ids` will not
        # change with the order of the `classes`
        self.cat_ids = self.coco.get_cat_ids(
            cat_names=self.metainfo['classes'])
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.cat_img_map = copy.deepcopy(self.coco.cat_img_map)

        img_ids = self.coco.get_img_ids()
        data_list = []
        total_ann_ids = []
        for img_id in img_ids:
            raw_img_info = self.coco.load_imgs([img_id])[0]
            raw_img_info['img_id'] = img_id

            ann_ids = self.coco.get_ann_ids(img_ids=[img_id])
            raw_ann_info = self.coco.load_anns(ann_ids)
            total_ann_ids.extend(ann_ids)

            parsed_data_info = self.parse_data_info({
                'raw_ann_info':
                raw_ann_info,
                'raw_img_info':
                raw_img_info
            })
            data_list.append(parsed_data_info)
        if self.ANN_ID_UNIQUE:
            assert len(set(total_ann_ids)) == len(
                total_ann_ids
            ), f"Annotation ids in '{self.ann_file}' are not unique!"

        del self.coco

        return data_list
    def parse_data_info(self, raw_data_info: dict) -> Union[dict, List[dict]]:
        """Parse raw annotation to target format.

        Args:
            raw_data_info (dict): Raw data information load from ``ann_file``

        Returns:
            Union[dict, List[dict]]: Parsed annotation.
        """
        img_info = raw_data_info['raw_img_info']
        ann_info = raw_data_info['raw_ann_info']

        data_info = {}

        # TODO: need to change data_prefix['img'] to data_prefix['img_path']
        img_path = osp.join(self.data_prefix['img'], img_info['file_name'])
        if self.data_prefix.get('seg', None):
            seg_map_path = osp.join(
                self.data_prefix['seg'],
                img_info['file_name'].rsplit('.', 1)[0] + self.seg_map_suffix)
        else:
            seg_map_path = None
        data_info['img_path'] = img_path
        data_info['img_id'] = img_info['img_id']
        data_info['seg_map_path'] = seg_map_path
        data_info['height'] = img_info['height']
        data_info['width'] = img_info['width']

        if self.return_classes:
            data_info['text'] = self.metainfo['classes']
            data_info['caption_prompt'] = self.caption_prompt
            data_info['custom_entities'] = True

        instances = []
        for i, ann in enumerate(ann_info):
            instance = {}

            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            if ann['category_id'] not in self.cat_ids:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]

            if ann.get('iscrowd', False):
                instance['ignore_flag'] = 1
            else:
                instance['ignore_flag'] = 0
            instance['bbox'] = bbox
            instance['bbox_label'] = self.cat2label[ann['category_id']]

            if ann.get('segmentation', None):
                instance['mask'] = ann['segmentation']

            instances.append(instance)
        data_info['instances'] = instances
        return data_info
    def __getitem__(self, idx: int) -> dict:
        """Get the idx-th image and data information of dataset after
        ``self.pipeline``, and ``full_init`` will be called if the dataset has
        not been fully initialized.

        During training phase, if ``self.pipeline`` get ``None``,
        ``self._rand_another`` will be called until a valid image is fetched or
         the maximum limit of refetech is reached.

        Args:
            idx (int): The index of self.data_list.

        Returns:
            dict: The idx-th image and data information of dataset after
            ``self.pipeline``.
        """
        # Performing full initialization by calling `__getitem__` will consume
        # extra memory. If a dataset is not fully initialized by setting
        # `lazy_init=True` and then fed into the dataloader. Different workers
        # will simultaneously read and parse the annotation. It will cost more
        # time and memory, although this may work. Therefore, it is recommended
        # to manually call `full_init` before dataset fed into dataloader to
        # ensure all workers use shared RAM from master process.
        if not self._fully_initialized:
#             print_log(
#                 'Please call `full_init()` method manually to accelerate '
#                 'the speed.',
#                 logger='current',
#                 level=logging.WARNING)
            self.full_init()

        if self.test_mode:
            data = self.prepare_data(idx)
            if data is None:
                raise Exception('Test time pipline should not get `None` '
                                'data_sample')
            return data

        for _ in range(self.max_refetch + 1):
            data = self.prepare_data(idx)
            # Broken images or random augmentations may cause the returned data
            # to be None
            if data is None:
                idx = self._rand_another()
                continue
            return data

        raise Exception(f'Cannot find valid image after {self.max_refetch}! '
                        'Please check your image path and pipeline')
    def filter_data(self) -> List[dict]:
        """Filter annotations according to filter_cfg.

        Returns:
            List[dict]: Filtered results.
        """
        if self.test_mode:
            return self.data_list

        if self.filter_cfg is None:
            return self.data_list

        filter_empty_gt = self.filter_cfg.get('filter_empty_gt', False)
        min_size = self.filter_cfg.get('min_size', 0)

        # obtain images that contain annotation
        ids_with_ann = set(data_info['img_id'] for data_info in self.data_list)
        # obtain images that contain annotations of the required categories
        ids_in_cat = set()
        for i, class_id in enumerate(self.cat_ids):
            ids_in_cat |= set(self.cat_img_map[class_id])
        # merge the image id sets of the two conditions and use the merged set
        # to filter out images if self.filter_empty_gt=True
        ids_in_cat &= ids_with_ann

        valid_data_infos = []
        for i, data_info in enumerate(self.data_list):
            img_id = data_info['img_id']
            width = data_info['width']
            height = data_info['height']
            if filter_empty_gt and img_id not in ids_in_cat:
                continue
            if min(width, height) >= min_size:
                valid_data_infos.append(data_info)

        return valid_data_infos
    def _serialize_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Serialize ``self.data_list`` to save memory when launching multiple
        workers in data loading. This function will be called in ``full_init``.

        Hold memory using serialized objects, and data loader workers can use
        shared RAM from master process instead of making a copy.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Serialized result and corresponding
            address.
        """

        def _serialize(data):
            buffer = pickle.dumps(data, protocol=4)
            return np.frombuffer(buffer, dtype=np.uint8)

        # Serialize data information list avoid making multiple copies of
        # `self.data_list` when iterate `import torch.utils.data.dataloader`
        # with multiple workers.
        data_list = [_serialize(x) for x in self.data_list]
        address_list = np.asarray([len(x) for x in data_list], dtype=np.int64)
        data_address: np.ndarray = np.cumsum(address_list)
        # TODO Check if np.concatenate is necessary
        data_bytes = np.concatenate(data_list)
        # Empty cache for preventing making multiple copies of
        # `self.data_info` when loading data multi-processes.
        self.data_list.clear()
        gc.collect()
        return data_bytes, data_address


# In[ ]:





# In[31]:


class Albu:
    #support albumentations type
    """Albumentation augmentation.

    Adds custom transformations from Albumentations library.
    Please, visit `https://albumentations.readthedocs.io`
    to get more information.

    Required Keys:

    - img (np.uint8)
    - gt_bboxes (HorizontalBoxes[torch.float32]) (optional)
    - gt_masks (BitmapMasks | PolygonMasks) (optional)

    Modified Keys:

    - img (np.uint8)
    - gt_bboxes (HorizontalBoxes[torch.float32]) (optional)
    - gt_masks (BitmapMasks | PolygonMasks) (optional)
    - img_shape (tuple)

    An example of ``transforms`` is as followed:

    .. code-block::

        [
            dict(
                type='ShiftScaleRotate',
                shift_limit=0.0625,
                scale_limit=0.0,
                rotate_limit=0,
                interpolation=1,
                p=0.5),
            dict(
                type='RandomBrightnessContrast',
                brightness_limit=[0.1, 0.3],
                contrast_limit=[0.1, 0.3],
                p=0.2),
            dict(type='ChannelShuffle', p=0.1),
            dict(
                type='OneOf',
                transforms=[
                    dict(type='Blur', blur_limit=3, p=1.0),
                    dict(type='MedianBlur', blur_limit=3, p=1.0)
                ],
                p=0.1),
        ]

    Args:
        transforms (list[dict]): A list of albu transformations
        bbox_params (dict, optional): Bbox_params for albumentation `Compose`
        keymap (dict, optional): Contains
            {'input key':'albumentation-style key'}
        skip_img_without_anno (bool): Whether to skip the image if no ann left
            after aug. Defaults to False.
    """

    def __init__(self,
                 transforms: List[dict],
                 bbox_params: Optional[dict] = None,
                 keymap: Optional[dict] = None,
                 skip_img_without_anno: bool = False) -> None:
#         if Compose is None:
#             raise RuntimeError('albumentations is not installed')

        # Args will be modified later, copying it will be safer
        transforms = copy.deepcopy(transforms)
        if bbox_params is not None:
            bbox_params = copy.deepcopy(bbox_params)
        if keymap is not None:
            keymap = copy.deepcopy(keymap)
        self.transforms = transforms
        self.filter_lost_elements = False
        self.skip_img_without_anno = skip_img_without_anno
#         assert all(transforms[i]['type'] in ['Blur','MedianBlur','ToGray','CLAHE'] for i in range(len(transforms)))
#         _transforms = []
#         for i in range(len(transforms)):
#             if transforms[i]['type'] == 'Blur':
#                 _transforms.append(Blur)
#             elif transforms[i]['type'] == 'MedianBlur':
#                 _transforms.append(MedianBlur)
#             elif transforms[i]['type'] == 'ToGray':
#                 _transforms.append(ToGray)
#             elif transforms[i]['type'] == 'CLAHE':
#                 _transforms.append(CLAHE)
#             else:
#                 raise NotImplementedError
#         self.aug = transform.Compose(_transforms)
        # A simple workaround to remove masks without boxes
        if (isinstance(bbox_params, dict) and 'label_fields' in bbox_params
                and 'filter_lost_elements' in bbox_params):
            self.filter_lost_elements = True
            self.origin_label_fields = bbox_params['label_fields']
            bbox_params['label_fields'] = ['idx_mapper']
            del bbox_params['filter_lost_elements']

        self.bbox_params = (
            self.albu_builder(bbox_params) if bbox_params else None)
        self.aug = Compose([self.albu_builder(t) for t in self.transforms],
                           bbox_params=self.bbox_params)
#             self.aug = transform.Compose()
        if not keymap:
            self.keymap_to_albu = {
                'img': 'image',
                'gt_masks': 'masks',
                'gt_bboxes': 'bboxes'
            }
        else:
            self.keymap_to_albu = keymap
        self.keymap_back = {v: k for k, v in self.keymap_to_albu.items()}

    def albu_builder(self, cfg: dict):
#         """Import a module from albumentations.

#         It inherits some of :func:`build_from_cfg` logic.

#         Args:
#             cfg (dict): Config dict. It should at least contain the key "type".

#         Returns:
#             obj: The constructed object.
#         """

        assert isinstance(cfg, dict) and 'type' in cfg
        args = cfg.copy()
        obj_type = args.pop('type')
#         assert obj_type ==
#         assert isinstance(obj_type,str)
        if isinstance(obj_type,str):
            obj_type = obj_type.lower()
#             if albumentations is None:
#                 raise RuntimeError('albumentations is not installed')
            obj_cls = albumentations[obj_type]
        elif inspect.isclass(obj_type):
            obj_cls = obj_type
        else:
            raise TypeError(
                f'type must be a str or valid type, but got {type(obj_type)}')

        if 'transforms' in args:
            args['transforms'] = [
                self.albu_builder(transform)
                for transform in args['transforms']
            ]

        return obj_cls(**args)

    
    @staticmethod
    def mapper(d: dict, keymap: dict) -> dict:
        """Dictionary mapper. Renames keys according to keymap provided.

        Args:
            d (dict): old dict
            keymap (dict): {'old_key':'new_key'}
        Returns:
            dict: new dict.
        """
        updated_dict = {}
        for k, v in zip(d.keys(), d.values()):
            new_k = keymap.get(k, k)
            updated_dict[new_k] = d[k]
        return updated_dict

#     @autocast_box_type()
    def __call__(self, results: dict) -> Union[dict, None]:
        """Transform function of Albu."""
        # TODO: gt_seg_map is not currently supported
        # dict to albumentations format
        results = self.mapper(results, self.keymap_to_albu)
        results, ori_masks = self._preprocess_results(results)
        results = self.aug(**results)
        results = self._postprocess_results(results, ori_masks)
        if results is None:
            return None
        # back to the original format
        results = self.mapper(results, self.keymap_back)
        results['img_shape'] = results['img'].shape[:2]
        return results

    def _preprocess_results(self, results: dict) -> tuple:
        """Pre-processing results to facilitate the use of Albu."""
        if 'bboxes' in results:
            # to list of boxes
#             if not isinstance(results['bboxes'], HorizontalBoxes):
#                 raise NotImplementedError(
#                     'Albu only supports horizontal boxes now')
            bboxes = np.array(results['bboxes'])
            results['bboxes'] = [x for x in bboxes]
            # add pseudo-field for filtration
            if self.filter_lost_elements:
                results['idx_mapper'] = np.arange(len(results['bboxes']))

        # TODO: Support mask structure in albu
        ori_masks = None
        if 'masks' in results:
            raise NotImplementedError('not suppport mask')
#             if isinstance(results['masks'], PolygonMasks):
#                 raise NotImplementedError(
#                     'Albu only supports BitMap masks now')
#             ori_masks = results['masks']
#             if albumentations.__version__ < '0.5':
#                 results['masks'] = results['masks'].masks
#             else:
#                 results['masks'] = [mask for mask in results['masks'].masks]

        return results, ori_masks

    def _postprocess_results(
            self,
            results: dict,
            ori_masks = None) -> dict:
        """Post-processing Albu output."""
        # albumentations may return np.array or list on different versions
        if 'gt_bboxes_labels' in results and isinstance(
                results['gt_bboxes_labels'], list):
            results['gt_bboxes_labels'] = np.array(
                results['gt_bboxes_labels'], dtype=np.int64)
        if 'gt_ignore_flags' in results and isinstance(
                results['gt_ignore_flags'], list):
            results['gt_ignore_flags'] = np.array(
                results['gt_ignore_flags'], dtype=bool)

        if 'bboxes' in results:
            if isinstance(results['bboxes'], list):
                results['bboxes'] = np.array(
                    results['bboxes'], dtype=np.float32)
            results['bboxes'] = results['bboxes'].reshape(-1, 4)
#             results['bboxes'] = HorizontalBoxes(results['bboxes'])

            # filter label_fields
            if self.filter_lost_elements:

                for label in self.origin_label_fields:
                    results[label] = np.array(
                        [results[label][i] for i in results['idx_mapper']])
                if 'masks' in results:
                    raise NotImplementedError
#                     assert ori_masks is not None
#                     results['masks'] = np.array(
#                         [results['masks'][i] for i in results['idx_mapper']])
#                     results['masks'] = ori_masks.__class__(
#                         results['masks'],
#                         results['masks'][0].shape[0],
#                         results['masks'][0].shape[1],
#                     )
#                 if (not len(results['idx_mapper'])
#                         and self.skip_img_without_anno):
#                     return None
#             elif 'masks' in results:
#                 results['masks'] = ori_masks.__class__(results['masks'],
#                                                        ori_masks.height,
#                                                        ori_masks.width)

        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__ + f'(transforms={self.transforms})'
        return repr_str


# In[ ]:





# In[32]:


class LetterResize:
    """Resize and pad image while meeting stride-multiple constraints.

    Required Keys:

    - img (np.uint8)
    - batch_shape (np.int64) (optional)

    Modified Keys:

    - img (np.uint8)
    - img_shape (tuple)
    - gt_bboxes (optional)

    Added Keys:
    - pad_param (np.float32)

    Args:
        scale (Union[int, Tuple[int, int]]): Images scales for resizing.
        pad_val (dict): Padding value. Defaults to dict(img=0, seg=255).
        use_mini_pad (bool): Whether using minimum rectangle padding.
            Defaults to True
        stretch_only (bool): Whether stretch to the specified size directly.
            Defaults to False
        allow_scale_up (bool): Allow scale up when ratio > 1. Defaults to True
        half_pad_param (bool): If set to True, left and right pad_param will
            be given by dividing padding_h by 2. If set to False, pad_param is
            in int format. We recommend setting this to False for object
            detection tasks, and True for instance segmentation tasks.
            Default to False.
    """

    def __init__(self,
                 scale: Union[int, Tuple[int, int]] = None,
                 pad_val: dict = dict(img=0, mask=0, seg=255),
                 use_mini_pad: bool = False,
                 stretch_only: bool = False,
                 allow_scale_up: bool = True,
                 half_pad_param: bool = False,
                 scale_factor: Optional[Union[float, Tuple[float,
                                                           float]]] = None,
                 keep_ratio: bool = True,
                 clip_object_border: bool = True,
                 backend: str = 'cv2',
                 interpolation='bilinear') -> None:
        assert scale is not None or scale_factor is not None, (
            '`scale` and'
            '`scale_factor` can not both be `None`')
        if scale is None:
            self.scale = None
        else:
            if isinstance(scale, int):
                self.scale = (scale, scale)
            else:
                self.scale = scale

        self.backend = backend
        self.interpolation = interpolation
        self.keep_ratio = keep_ratio
        self.clip_object_border = clip_object_border
        if scale_factor is None:
            self.scale_factor = None
        elif isinstance(scale_factor, float):
            self.scale_factor = (scale_factor, scale_factor)
        elif isinstance(scale_factor, tuple):
            assert (len(scale_factor)) == 2
            self.scale_factor = scale_factor
        else:
            raise TypeError(
                f'expect scale_factor is float or Tuple(float), but'
                f'get {type(scale_factor)}')
        self.pad_val = pad_val
        if isinstance(pad_val, (int, float)):
            pad_val = dict(img=pad_val, seg=255)
        assert isinstance(
            pad_val, dict), f'pad_val must be dict, but got {type(pad_val)}'

        self.use_mini_pad = use_mini_pad
        self.stretch_only = stretch_only
        self.allow_scale_up = allow_scale_up
        self.half_pad_param = half_pad_param

    def _resize_img(self, results: dict):
        """Resize images with ``results['scale']``."""
        image = results.get('img', None)
        if image is None:
            return

        # Use batch_shape if a batch_shape policy is configured
        if 'batch_shape' in results:
            scale = tuple(results['batch_shape'])  # hw
        else:
            scale = self.scale[::-1]  # wh -> hw

        image_shape = image.shape[:2]  # height, width

        # Scale ratio (new / old)
        ratio = min(scale[0] / image_shape[0], scale[1] / image_shape[1])

        # only scale down, do not scale up (for better test mAP)
        if not self.allow_scale_up:
            ratio = min(ratio, 1.0)

        ratio = [ratio, ratio]  # float -> (float, float) for (height, width)

        # compute the best size of the image
        no_pad_shape = (int(round(image_shape[0] * ratio[0])),
                        int(round(image_shape[1] * ratio[1])))

        # padding height & width
        padding_h, padding_w = [
            scale[0] - no_pad_shape[0], scale[1] - no_pad_shape[1]
        ]
        if self.use_mini_pad:
            # minimum rectangle padding
            padding_w, padding_h = np.mod(padding_w, 32), np.mod(padding_h, 32)

        elif self.stretch_only:
            # stretch to the specified size directly
            padding_h, padding_w = 0.0, 0.0
            no_pad_shape = (scale[0], scale[1])
            ratio = [scale[0] / image_shape[0],
                     scale[1] / image_shape[1]]  # height, width ratios

        if image_shape != no_pad_shape:
            # compare with no resize and padding size
            image = (transform.resize(
                img = image, size = (no_pad_shape[0], no_pad_shape[1]),
                interpolation=self.interpolation,
                )).numpy()

        scale_factor = (no_pad_shape[1] / image_shape[1],
                        no_pad_shape[0] / image_shape[0])

        if 'scale_factor' in results:
            results['scale_factor_origin'] = results['scale_factor']
        results['scale_factor'] = scale_factor

        # padding
        top_padding, left_padding = int(round(padding_h // 2 - 0.1)), int(
            round(padding_w // 2 - 0.1))
        bottom_padding = padding_h - top_padding
        right_padding = padding_w - left_padding

        padding_list = [
            top_padding, bottom_padding, left_padding, right_padding
        ]
        if top_padding != 0 or bottom_padding != 0 or                 left_padding != 0 or right_padding != 0:

            pad_val = self.pad_val.get('img', 0)
            if isinstance(pad_val, int) and image.ndim == 3:
                pad_val = tuple(pad_val for _ in range(image.shape[2]))

            image = impad(
                img=image,
                padding=(padding_list[2], padding_list[0], padding_list[3],
                         padding_list[1]),
                pad_val=pad_val,
                padding_mode='constant')

        results['img'] = image
        results['img_shape'] = image.shape
        if 'pad_param' in results:
            results['pad_param_origin'] = results['pad_param'] *                                           np.repeat(ratio, 2)

        if self.half_pad_param:
            results['pad_param'] = np.array(
                [padding_h / 2, padding_h / 2, padding_w / 2, padding_w / 2],
                dtype=np.float32)
        else:
            # We found in object detection, using padding list with
            # int type can get higher mAP.
            results['pad_param'] = np.array(padding_list, dtype=np.float32)

    def _resize_masks(self, results: dict):
        """Resize masks with ``results['scale']``"""
        if results.get('gt_masks', None) is None:
            return

        gt_masks = results['gt_masks']
        assert isinstance(
            gt_masks, PolygonMasks
        ), f'Only supports PolygonMasks, but got {type(gt_masks)}'

        # resize the gt_masks
        gt_mask_h = results['gt_masks'].height * results['scale_factor'][1]
        gt_mask_w = results['gt_masks'].width * results['scale_factor'][0]
        gt_masks = transform.resize(img = results['gt_masks'],
            shape = (int(round(gt_mask_h)), int(round(gt_mask_w))))
        gt_masks.numpy()
        top_padding, _, left_padding, _ = results['pad_param']
        if int(left_padding) != 0:
            gt_masks = mask_translate(gt_masks,
                out_shape=results['img_shape'][:2],
                offset=int(left_padding),
                direction='horizontal')
        if int(top_padding) != 0:
            gt_masks = mask_translate(gt_masks,
                out_shape=results['img_shape'][:2],
                offset=int(top_padding),
                direction='vertical')
        results['gt_masks'] = gt_masks

    def _resize_bboxes(self, results: dict):
        """Resize bounding boxes with ``results['scale_factor']``."""
        if results.get('gt_bboxes', None) is None:
            return
        bbox_rescale(results['gt_bboxes'],results['scale_factor'])

        if len(results['pad_param']) != 4:
            return
        bbox.translate(results['gt_bboxes'],
            (results['pad_param'][2], results['pad_param'][0]))

        if self.clip_object_border:
            bbox.clip(results['gt_bboxes'], (results['img_shape'][1],results['img_shape'][0]))

    def __call__(self, results: dict) -> dict:
        results = super().transform(results)
        if 'scale_factor_origin' in results:
            scale_factor_origin = results.pop('scale_factor_origin')
            results['scale_factor'] = (results['scale_factor'][0] *
                                       scale_factor_origin[0],
                                       results['scale_factor'][1] *
                                       scale_factor_origin[1])
        if 'pad_param_origin' in results:
            pad_param_origin = results.pop('pad_param_origin')
            results['pad_param'] += pad_param_origin
        return results


# In[ ]:





# In[33]:


class YOLOv5KeepRatioResize:  
    def __init__(self,
                 scale: Optional[Union[int, Tuple[int, int]]] = None,
                 scale_factor: Optional[Union[float, Tuple[float,
                                                           float]]] = None,
                 keep_ratio: bool = True,
                 clip_object_border: bool = True,
                 backend: str = None,
                 interpolation='bilinear') -> None:
        assert scale is not None or scale_factor is not None, (
            '`scale` and'
            '`scale_factor` can not both be `None`')
        if scale is None:
            self.scale = None
        else:
            if isinstance(scale, int):
                self.scale = (scale, scale)
            else:
                self.scale = scale

        self.backend = backend
        self.interpolation = interpolation
        self.keep_ratio = keep_ratio
        self.clip_object_border = clip_object_border
        if scale_factor is None:
            self.scale_factor = None
        elif isinstance(scale_factor, float):
            self.scale_factor = (scale_factor, scale_factor)
        elif isinstance(scale_factor, tuple):
            assert (len(scale_factor)) == 2
            self.scale_factor = scale_factor
        else:
            raise TypeError(
                f'expect scale_factor is float or Tuple(float), but'
                f'get {type(scale_factor)}')

    def _get_rescale_ratio(old_size: Tuple[int, int],
                           scale: Union[float, Tuple[int]]) -> float:
        
        """Calculate the ratio for rescaling.

        Args:
            old_size (tuple[int]): The old size (w, h) of image.
            scale (float | tuple[int]): The scaling factor or maximum size.
                If it is a float number, then the image will be rescaled by
                this factor, else if it is a tuple of 2 integers, then
                the image will be rescaled as large as possible within
                the scale.

        Returns:
            float: The resize ratio.
        """
        w, h = old_size
        if isinstance(scale, (float, int)):
            if scale <= 0:
                raise ValueError(f'Invalid scale {scale}, must be positive.')
            scale_factor = scale
        elif isinstance(scale, tuple):
            max_long_edge = max(scale)
            max_short_edge = min(scale)
            scale_factor = min(max_long_edge / max(w, h),
                               max_short_edge / min(w, h))
        else:
            raise TypeError('Scale must be a number or tuple of int, '
                            f'but got {type(scale)}')

        return scale_factor

    def _resize_img(self, results: dict):
        """Resize images with ``results['scale']``."""
        assert self.keep_ratio is True

        if results.get('img', None) is not None:
            image = results['img']
            original_h, original_w = image.shape[:2]
            ratio = self._get_rescale_ratio((original_w, original_h),
                                            self.scale)

            if ratio != 1:
#                 image = mmcv.imresize(
#                     img=image,
#                     size=(int(original_w * ratio), int(original_h * ratio)),
#                     interpolation='area' if ratio < 1 else 'bilinear',
#                     backend=self.backend)
                image = transform.resize(
                    img=image,
                    size=(int(original_h * ratio), int(original_w * ratio)),
                    interpolation='area' if ratio < 1 else 'bilinear',
                )

            resized_h, resized_w = image.shape[:2]
            scale_ratio_h = resized_h / original_h
            scale_ratio_w = resized_w / original_w
            scale_factor = (scale_ratio_w, scale_ratio_h)
            image = image.numpy()
            results['img'] = image
            results['img_shape'] = image.shape[:2]
            results['scale_factor'] = scale_factor
    def _resize_masks(self, results: dict) -> None:
        """Resize masks with ``results['scale']``"""
        if results.get('gt_masks', None) is not None:
            if self.keep_ratio:
                
#                 results['gt_masks'] = results['gt_masks'].rescale(
#                     results['scale'])
                results['gt_masks'] = bbox_rescale(results['gt_masks'],results['scale'])
            else:
#                 results['gt_masks'] = results['gt_masks'].resize(
#                     results['img_shape'])

                result['gt_masks'] = (transform.resize(img=results['gt_masks'],
                                                       size=results['img_shape'])).numpy()
                                      
    def _resize_bboxes(self, results: dict) -> None:
        """Resize bounding boxes with ``results['scale_factor']``."""
        if results.get('gt_bboxes', None) is not None:
            results['gt_bboxes'] = bbox_rescale(results['gt_bboxes'], results['scale_factor'])
#             results['gt_bboxes'].rescale_(results['scale_factor'])
            if self.clip_object_border:
                results['gt_bboxes'] = bbox_clip(results['gt_bboxes'], results['img_shape'])
#                 results['gt_bboxes'].clip_(results['img_shape'])

    def _record_homography_matrix(self, results: dict) -> None:
        """Record the homography matrix for the Resize."""
        w_scale, h_scale = results['scale_factor'] # TODO1 scale_factor in results should be w,h, some origin code use hw,see if need change
        homography_matrix = np.array(
            [[w_scale, 0, 0], [0, h_scale, 0], [0, 0, 1]], dtype=np.float32)
        if results.get('homography_matrix', None) is None:
            results['homography_matrix'] = homography_matrix
        else:
            results['homography_matrix'] = homography_matrix @ results[
                'homography_matrix']

    def _resize_seg(self, results: dict) -> None:
        """Resize semantic segmentation map with ``results['scale']``."""
        if results.get('gt_seg_map', None) is not None:
            if self.keep_ratio:
#                 gt_seg = mmcv.imrescale(
#                     results['gt_seg_map'],
#                     results['scale'],
#                     interpolation='nearest',
#                     backend=self.backend)
                gt_seg = imrescale(
                    img = results['gt_seg_map'],
                    size = results['scale'],
                    interpolation='nearest',
                )
            else:
#                 gt_seg = mmcv.imresize(
#                     results['gt_seg_map'],
#                     results['scale'],
#                     interpolation='nearest',
#                     backend=self.backend)
                gt_seg = (transform.resize(
                    img = results['gt_seg_map'],
                    size = (results['scale'][1],results['scale'][0]), #change
                    interpolation='nearest',
                )).numpy()
            results['gt_seg_map'] = gt_seg

    def _resize_keypoints(self, results: dict) -> None:
        """Resize keypoints with ``results['scale_factor']``."""
        if results.get('gt_keypoints', None) is not None:
            keypoints = results['gt_keypoints']

            keypoints[:, :, :2] = keypoints[:, :, :2] * np.array(
                results['scale_factor'])
            if self.clip_object_border:
                keypoints[:, :, 0] = np.clip(keypoints[:, :, 0], 0,
                                             results['img_shape'][1])
                keypoints[:, :, 1] = np.clip(keypoints[:, :, 1], 0,
                                             results['img_shape'][0])
            results['gt_keypoints'] = keypoints
    def __call__(self, results: dict) -> dict:
        """Transform function to resize images, bounding boxes and semantic
        segmentation map.

        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Resized results, 'img', 'gt_bboxes', 'gt_seg_map',
            'scale', 'scale_factor', 'height', 'width', and 'keep_ratio' keys
            are updated in result dict.
        """
        if self.scale:
            results['scale'] = self.scale
        else:
            img_shape = results['img'].shape[:2]
            results['scale'] = _scale_size(img_shape[::-1], self.scale_factor)
        self._resize_img(results)
        self._resize_bboxes(results)
        self._resize_masks(results)
        self._resize_seg(results)
        self._record_homography_matrix(results)
        return results
    



# In[ ]:





# In[34]:


class YOLOv5HSVRandomAug:
    """Apply HSV augmentation to image sequentially.

    Required Keys:

    - img

    Modified Keys:

    - img

    Args:
        hue_delta ([int, float]): delta of hue. Defaults to 0.015.
        saturation_delta ([int, float]): delta of saturation. Defaults to 0.7.
        value_delta ([int, float]): delta of value. Defaults to 0.4.
    """

    def __init__(self,
                 hue_delta: Union[int, float] = 0.015,
                 saturation_delta: Union[int, float] = 0.7,
                 value_delta: Union[int, float] = 0.4):
        self.hue_delta = hue_delta
        self.saturation_delta = saturation_delta
        self.value_delta = value_delta

    def __call__(self, results: dict) -> dict:
        """The HSV augmentation transform function.

        Args:
            results (dict): The result dict.

        Returns:
            dict: The result dict.
        """
        hsv_gains = np.array(            random.uniform(-1, 1, 3) *             [self.hue_delta, self.saturation_delta, self.value_delta] + 1)
#         hue, sat, val = cv2.split(
#             cv2.cvtColor(results['img'], cv2.COLOR_BGR2HSV))
        img = Image.fromarray(results['img']) # TODO : results[img] can not be Image object must np.ndarray
        hsv = img.convert('HSV')
        hsv_arr = np.array(hsv)
        h_channel = ((hsv_arr[..., 0].astype(np.float32) * hsv_gains[0]) % 256).astype(np.unit8)
        s_channel = np.clip((hsv_arr[..., 1].astype(np.float32) * hsv_gains[1]), 0, 255).astype(np.uint8)
        v_channel = np.clip((hsv_arr[..., 2].astype(np.float32) * hsv_gains[2]), 0, 255).astype(np.uint8)
        hsv_obj = np.stack[h_channel, s_channel, v_channel]
        image = Image.fromarray(hsv_obj,mode='HSV')
        results['img'] = np.array(image.convert('RGB'))
#         table_list = np.arange(0, 256, dtype=hsv_gains.dtype)
#         lut_hue = ((table_list * hsv_gains[0]) % 180).astype(np.uint8)
#         lut_sat = np.clip(table_list * hsv_gains[1], 0, 255).astype(np.uint8)
#         lut_val = np.clip(table_list * hsv_gains[2], 0, 255).astype(np.uint8)

#         im_hsv = cv2.merge(
#             (cv2.LUT(hue, lut_hue), cv2.LUT(sat,
#                                             lut_sat), cv2.LUT(val, lut_val)))
#         results['img'] = cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR)
        return results


# In[ ]:





# In[35]:


class YOLOv5RandomAffine:
    def __init__(self,
                 max_rotate_degree: float = 10.0,
                 max_translate_ratio: float = 0.1,
                 scaling_ratio_range: Tuple[float, float] = (0.5, 1.5),
                 max_shear_degree: float = 2.0,
                 border: Tuple[int, int] = (0, 0),
                 border_val: Tuple[int, int, int] = (114, 114, 114),
                 bbox_clip_border: bool = True,
                 min_bbox_size: int = 2,
                 min_area_ratio: float = 0.1,
                 use_mask_refine: bool = False,
                 max_aspect_ratio: float = 20.,
                 resample_num: int = 1000) -> None:
        assert 0 <= max_translate_ratio <= 1
        assert scaling_ratio_range[0] <= scaling_ratio_range[1]
        assert scaling_ratio_range[0] > 0
        self.max_rotate_degree = max_rotate_degree
        self.max_translate_ratio = max_translate_ratio
        self.scaling_ratio_range = scaling_ratio_range
        self.max_shear_degree = max_shear_degree
        self.border = border
        self.border_val = border_val
        self.bbox_clip_border = bbox_clip_border
        self.min_bbox_size = min_bbox_size
        self.min_area_ratio = min_area_ratio
        # The use_mask_refine parameter has been deprecated.
        self.use_mask_refine = use_mask_refine
        self.max_aspect_ratio = max_aspect_ratio
        self.resample_num = resample_num
        
    def bbox_project(bbox,w_matrix): # TODO if for compatibility can replace jt with np
        bboxes = jt.array(bbox)
        wrap_matrix = jt.array(w_matrix)
        x1, y1, x2, y2 = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]
        ver = jt.stack([
            jt.stack(x1, y1, dim=1),
            jt.stack(x2, y1, dim=1),
            jt.stack(x1, y2, dim=1),
            jt.stack(x2, y2, dim=1),
        ])
        h_var = jt.concat([
            vertices,
            jt.ones((
                ver.shape[0],
                ver.shape[1],
                1
            ),dim = 2)
        ])
        t_h = jt.enisum('ij,nki -> nki',wrap_matrix,bboxes)
        w = t_h[..., 2:3]
        t_v = t_h[..., :2] / w
        n_x1 = t_v[...,0].min(dim=1)
        n_y1 = t_v[...,1].min(dim=1)
        n_x2 = t_v[...,0].max(dim=1)
        n_y2 = t_v[...,1].max(dim = 1)
        
        bboxes = (jt.stack([n_x1,n_y1,n_x2,n_y2])).numpy
        return bboxes
    
    
    def bbox_is_inside(bboxes,area:list):
        
        width ,height = enumerate(area)
        cond1 = bboxes[...,0] >=0
        cond2 = bboxes[...,1] >=0
        cond3 = bboxes[...,2] <= width
        cond4 = bboxes[...,3] <= height
        return cond1 and cond2 and cond3 and cond4
        
    def __call__(self, result : dict) -> dict:
        img = results['img']
        height = img.shape[0] + self.border[1] * 2
        width = img.shape[1] + self.border[0] * 2

        warp_matrix = self._get_random_homography_matrix(width, height)
        i = warp_matrix[2,2]
        if abs(i) < 1e-6:
            raise ValueError
        n_wrap_matrix = warp_matrix / i
        a, b, c = n_wrap_matrix[0]
        d,e,f = n_wrap_matrix[1]
        g,h,_ = n_wrap_matrix[2]
        per_data = (a,b,c,d,e,f,g,h) 

#         img = cv2.warpPerspective(
#             img,
#             warp_matrix,
#             dsize=(width, height),
#             borderValue=self.border_val)
        img = Image.fromarray(img)
        img = img.transform(
            size = (width, height),
            method = Image.PERSPECTIVE,
            data = per_data,
            resample = Image.BILINEAR # NOTICE: possible difference source
        )
        if self.border_val != (0,0,0):
            bg = Image.new("RGB",(width,height),self.border_val)
            bg.paste(img)
            img = bg
        img = np.array(img)
        results['img'] = img
        results['img_shape'] = img.shape

        bboxes = results['gt_bboxes'] # TODO : should get [x1,y1,x2,y2]*N
        num_bboxes = len(bboxes)
        if num_bboxes:
            bbox_project(bboxes,warp_matrix)
            if self.bbox_clip_border:
                bbox_clip(bboxes,[width, height])
            # remove outside bbox
            valid_index = bbox_is_inside(bbox,[width, height]).numpy()
            results['gt_bboxes'] = bboxes[valid_index]
            results['gt_bboxes_labels'] = results['gt_bboxes_labels'][
                valid_index]
            results['gt_ignore_flags'] = results['gt_ignore_flags'][
                valid_index]

#             if 'gt_masks' in results:
#                 raise NotImplementedError('RandomAffine only supports bbox.')

#             if 'gt_keypoints' in results:
#                 keypoints = results['gt_keypoints']
#                 keypoints.project_(warp_matrix)
#                 if self.bbox_clip_border:
#                     keypoints.clip_([height, width])
#                 results['gt_keypoints'] = keypoints[valid_index]

        return results
    
    # get prj_matrix
    
    def _get_random_homography_matrix(self, width, height):
        # Rotation
        rotation_degree = random.uniform(-self.max_rotate_degree,
                                         self.max_rotate_degree)
        rotation_matrix = self._get_rotation_matrix(rotation_degree)

        # Scaling
        scaling_ratio = random.uniform(self.scaling_ratio_range[0],
                                       self.scaling_ratio_range[1])
        scaling_matrix = self._get_scaling_matrix(scaling_ratio)

        # Shear
        x_degree = random.uniform(-self.max_shear_degree,
                                  self.max_shear_degree)
        y_degree = random.uniform(-self.max_shear_degree,
                                  self.max_shear_degree)
        shear_matrix = self._get_shear_matrix(x_degree, y_degree)

        # Translation
        trans_x = random.uniform(-self.max_translate_ratio,
                                 self.max_translate_ratio) * width
        trans_y = random.uniform(-self.max_translate_ratio,
                                 self.max_translate_ratio) * height
        translate_matrix = self._get_translation_matrix(trans_x, trans_y)

        warp_matrix = (
                translate_matrix @ shear_matrix @ rotation_matrix @ scaling_matrix)
        return warp_matrix
    def _get_rotation_matrix(rotate_degrees: float) -> np.ndarray:
        radian = math.radians(rotate_degrees)
        rotation_matrix = np.array(
            [[np.cos(radian), -np.sin(radian), 0.],
             [np.sin(radian), np.cos(radian), 0.], [0., 0., 1.]],
            dtype=np.float32)
        return rotation_matrix

    @staticmethod
    def _get_scaling_matrix(scale_ratio: float) -> np.ndarray:
        scaling_matrix = np.array(
            [[scale_ratio, 0., 0.], [0., scale_ratio, 0.], [0., 0., 1.]],
            dtype=np.float32)
        return scaling_matrix

    @staticmethod
    def _get_shear_matrix(x_shear_degrees: float,
                          y_shear_degrees: float) -> np.ndarray:
        x_radian = math.radians(x_shear_degrees)
        y_radian = math.radians(y_shear_degrees)
        shear_matrix = np.array([[1, np.tan(x_radian), 0.],
                                 [np.tan(y_radian), 1, 0.], [0., 0., 1.]],
                                dtype=np.float32)
        return shear_matrix

    @staticmethod
    def _get_translation_matrix(x: float, y: float) -> np.ndarray:
        translation_matrix = np.array([[1, 0., x], [0., 1, y], [0., 0., 1.]],
                                      dtype=np.float32)
        return translation_matrix
    
        


# In[ ]:





# In[36]:


class YOLO_Mosaic:
    """Mosaic augmentation.

    Given 4 images, mosaic transform combines them into
    one output image. The output image is composed of the parts from each sub-
    image.

    .. code:: text

                        mosaic transform
                           center_x
                +------------------------------+
                |       pad        |           |
                |      +-----------+    pad    |
                |      |           |           |
                |      |  image1   +-----------+
                |      |           |           |
                |      |           |   image2  |
     center_y   |----+-+-----------+-----------+
                |    |   cropped   |           |
                |pad |   image3    |   image4  |
                |    |             |           |
                +----|-------------+-----------+
                     |             |
                     +-------------+

     The mosaic transform steps are as follows:

         1. Choose the mosaic center as the intersections of 4 images
         2. Get the left top image according to the index, and randomly
            sample another 3 images from the custom dataset.
         3. Sub image will be cropped if image is larger than mosaic patch

    Required Keys:

    - img
    - gt_bboxes (BaseBoxes[torch.float32]) (optional)
    - gt_bboxes_labels (np.int64) (optional)
    - gt_ignore_flags (bool) (optional)
    - mix_results (List[dict])

    Modified Keys:

    - img
    - img_shape
    - gt_bboxes (optional)
    - gt_bboxes_labels (optional)
    - gt_ignore_flags (optional)

    Args:
        img_scale (Sequence[int]): Image size after mosaic pipeline of single
            image. The shape order should be (width, height).
            Defaults to (640, 640).
        center_ratio_range (Sequence[float]): Center ratio range of mosaic
            output. Defaults to (0.5, 1.5).
        bbox_clip_border (bool, optional): Whether to clip the objects outside
            the border of the image. In some dataset like MOT17, the gt bboxes
            are allowed to cross the border of images. Therefore, we don't
            need to clip the gt bboxes in these cases. Defaults to True.
        pad_val (int): Pad value. Defaults to 114.
        pre_transform(Sequence[dict]): Sequence of transform object or
            config dict to be composed.
        prob (float): Probability of applying this transformation.
            Defaults to 1.0.
        use_cached (bool): Whether to use cache. Defaults to False.
        max_cached_images (int): The maximum length of the cache. The larger
            the cache, the stronger the randomness of this transform. As a
            rule of thumb, providing 10 caches for each image suffices for
            randomness. Defaults to 40.
        random_pop (bool): Whether to randomly pop a result from the cache
            when the cache is full. If set to False, use FIFO popping method.
            Defaults to True.
        max_refetch (int): The maximum number of retry iterations for getting
            valid results from the pipeline. If the number of iterations is
            greater than `max_refetch`, but results is still None, then the
            iteration is terminated and raise the error. Defaults to 15.
    """

    def __init__(self,
                 img_scale: Tuple[int, int] = (640, 640),
                 center_ratio_range: Tuple[float, float] = (0.5, 1.5),
                 bbox_clip_border: bool = True,
                 pad_val: float = 114.0,
                 pre_transform: Sequence[dict] = None,
                 prob: float = 1.0,
                 use_cached: bool = False,
                 max_cached_images: int = 40,
                 random_pop: bool = True,
                 max_refetch: int = 15):
        assert isinstance(img_scale, tuple)
        assert 0 <= prob <= 1.0, 'The probability should be in range [0,1]. '                                  f'got {prob}.'
        if use_cached:
            assert max_cached_images >= 4, 'The length of cache must >= 4, '                                            f'but got {max_cached_images}.'
        self.max_refetch = max_refetch
        self.prob = prob

        self.use_cached = use_cached
        self.max_cached_images = max_cached_images
        self.random_pop = random_pop
        self.results_cache = []

        if pre_transform is None:
            self.pre_transform = None
        else:
            self.pre_transform = transform.Compose(pre_transform)
        self.img_scale = img_scale
        self.center_ratio_range = center_ratio_range
        self.bbox_clip_border = bbox_clip_border
        self.pad_val = pad_val
    def get_indexes(self, dataset: Union) -> list:
        """Call function to collect indexes.

        Args:
            dataset (:obj:`Dataset` or list): The dataset or cached list.

        Returns:
            list: indexes.
        """
        indexes = [random.randint(0, len(dataset)) for _ in range(3)]
        return indexes

    def mix_img_transform(self, results: dict) -> dict:
        """Mixed image data transformation.

        Args:
            results (dict): Result dict.

        Returns:
            results (dict): Updated result dict.
        """
        assert 'mix_results' in results
        mosaic_bboxes = []
        mosaic_bboxes_labels = []
        mosaic_ignore_flags = []
        mosaic_masks = []
        mosaic_kps = []
        with_mask = True if 'gt_masks' in results else False
        with_kps = True if 'gt_keypoints' in results else False
        # self.img_scale is wh format # TODO : imge_scale = (width,height) change for aompatible
        img_scale_w, img_scale_h = self.img_scale

        if len(results['img'].shape) == 3:
            mosaic_img = np.full(
                (int(img_scale_h * 2), int(img_scale_w * 2), 3),
                self.pad_val,
                dtype=results['img'].dtype)
        else:
            mosaic_img = np.full((int(img_scale_h * 2), int(img_scale_w * 2)),
                                 self.pad_val,
                                 dtype=results['img'].dtype)

        # mosaic center x, y
        center_x = int(random.uniform(*self.center_ratio_range) * img_scale_w)
        center_y = int(random.uniform(*self.center_ratio_range) * img_scale_h)
        center_position = (center_x, center_y)

        loc_strs = ('top_left', 'top_right', 'bottom_left', 'bottom_right')
        for i, loc in enumerate(loc_strs):
            if loc == 'top_left':
                results_patch = results
            else:
                results_patch = results['mix_results'][i - 1]

            img_i = results_patch['img']
            h_i, w_i = img_i.shape[:2]
            # keep_ratio resize
            scale_ratio_i = min(img_scale_h / h_i, img_scale_w / w_i)
#             img_i = mmcv.imresize(
#                 img_i, (int(w_i * scale_ratio_i), int(h_i * scale_ratio_i)))
            img_i = transform.resize(
                img = img_i,
                size = (int(h_i * scale_ratio_i), int(w_i * scale_ratio_i))
            )

            # compute the combine parameters
            paste_coord, crop_coord = self._mosaic_combine(
                loc, center_position, img_i.shape[:2][::-1])
            x1_p, y1_p, x2_p, y2_p = paste_coord
            x1_c, y1_c, x2_c, y2_c = crop_coord

            # crop and paste image
            mosaic_img[y1_p:y2_p, x1_p:x2_p] = img_i[y1_c:y2_c, x1_c:x2_c]

            # adjust coordinate
            gt_bboxes_i = results_patch['gt_bboxes']
            gt_bboxes_labels_i = results_patch['gt_bboxes_labels']
            gt_ignore_flags_i = results_patch['gt_ignore_flags']

            padw = x1_p - x1_c
            padh = y1_p - y1_c
#             gt_bboxes_i.rescale_([scale_ratio_i, scale_ratio_i])
            gt_bboxes_i = bbox_rescale(gt_bboxes_i, [scale_ratio_i, scale_ratio_i])
#             gt_bboxes_i.translate_([padw, padh])
            gt_bboxes_i = bbox_translate(gt_bboxes_i, [padw, padh])
            mosaic_bboxes.append(gt_bboxes_i)
            mosaic_bboxes_labels.append(gt_bboxes_labels_i)
            mosaic_ignore_flags.append(gt_ignore_flags_i)
            if with_mask and results_patch.get('gt_masks', None) is not None:
                gt_masks_i = results_patch['gt_masks']
#                 gt_masks_i = gt_masks_i.resize(img_i.shape[:2])
#t odo whether we can treat mask as img yes
                gt_masks_i = transform.resize(img = gt_masks_i, size = img_i.shape[:2], interpolation = 'nearest')
#                 gt_masks_i = gt_masks_i.translate(
#                     out_shape=(int(self.img_scale[0] * 2),
#                                int(self.img_scale[1] * 2)),
#                     offset=padw,
#                     direction='horizontal')
#                 gt_masks_i = gt_masks_i.translate(
#                     out_shape=(int(self.img_scale[0] * 2),
#                                int(self.img_scale[1] * 2)),
#                     offset=padh,
#                     direction='vertical')
                gt_masks_i = mask_translate(
                    out_shape=(int(self.img_scale[0] * 2),
                    int(self.img_scale[1] * 2)),
                    offset=padw,
                    direction='horizontal'
                )
                gt_masks_i = mask_translate(
                    out_shape=(int(self.img_scale[0] * 2),
                    int(self.img_scale[1] * 2)),
                    offset=padh,
                    direction='vertical'
                )
                mosaic_masks.append(gt_masks_i)
            if with_kps and results_patch.get('gt_keypoints',
                                              None) is not None:
                gt_kps_i = results_patch['gt_keypoints']
#                 gt_kps_i.rescale_([scale_ratio_i, scale_ratio_i])
#                 gt_kps_i.translate_([padw, padh])
#Note:kp method like bbox
                gt_kps_i = bbox_rescale(gt_kps_i, [scale_ratio_i, scale_ratio_i])
                gt_kps_i = bbox_translate(gt_kps_i, [padw, padh])
                mosaic_kps.append(gt_kps_i)

        mosaic_bboxes = mosaic_bboxes[0].cat(mosaic_bboxes, 0)
        mosaic_bboxes_labels = np.concatenate(mosaic_bboxes_labels, 0)
        mosaic_ignore_flags = np.concatenate(mosaic_ignore_flags, 0)

        if self.bbox_clip_border:
            bbox_clip(mosaic_bboxes, [2 * img_scale_w, 2 * img_scale_h])
            if with_mask:
                mosaic_masks = mosaic_masks[0].cat(mosaic_masks)
                results['gt_masks'] = mosaic_masks
            if with_kps:
                mosaic_kps = mosaic_kps[0].cat(mosaic_kps, 0)
                bbox_clip(mosaic_kps, [2 * img_scale_w, 2 * img_scale_h])
                results['gt_keypoints'] = mosaic_kps
        else:
            # remove outside bboxes
            inside_inds = bbox_is_inside(
                mosaic_bboxes, [2 * img_scale_w, 2 * img_scale_h]).numpy()
            mosaic_bboxes = mosaic_bboxes[inside_inds]
            mosaic_bboxes_labels = mosaic_bboxes_labels[inside_inds]
            mosaic_ignore_flags = mosaic_ignore_flags[inside_inds]
#             if with_mask:
#                 mosaic_masks = mosaic_masks[0].cat(mosaic_masks)[inside_inds]
#                 results['gt_masks'] = mosaic_masks
# not understand how cat_fun goes,if something goes wrong need to check TODO
#             if with_kps:
#                 mosaic_kps = mosaic_kps[0].cat(mosaic_kps, 0)
#                 mosaic_kps = mosaic_kps[inside_inds]
#                 results['gt_keypoints'] = mosaic_kps
            if with_mask or with_kps:
                raise TypeError("unsuuported type mask in Mosica,need to get to know cat_fun")

        results['img'] = mosaic_img
        results['img_shape'] = mosaic_img.shape
        results['gt_bboxes'] = mosaic_bboxes
        results['gt_bboxes_labels'] = mosaic_bboxes_labels
        results['gt_ignore_flags'] = mosaic_ignore_flags

        return results

#     def cat(cls: Type[T], masks: Sequence[T]) -> T:
#         """Concatenate a sequence of masks into one single mask instance.

#         Args:
#             masks (Sequence[BitmapMasks]): A sequence of mask instances.

#         Returns:
#             BitmapMasks: Concatenated mask instance.
#         """
#         assert isinstance(masks, Sequence)
#         if len(masks) == 0:
#             raise ValueError('masks should not be an empty list.')
#         assert all(isinstance(m, cls) for m in masks)

#         mask_array = np.concatenate([m.masks for m in masks], axis=0)
#         return cls(mask_array, *mask_array.shape[1:])

    def _mosaic_combine(
            self, loc: str, center_position_xy: Sequence[float],
            img_shape_wh: Sequence[int]) -> Tuple[Tuple[int], Tuple[int]]:
        """Calculate global coordinate of mosaic image and local coordinate of
        cropped sub-image.

        Args:
            loc (str): Index for the sub-image, loc in ('top_left',
              'top_right', 'bottom_left', 'bottom_right').
            center_position_xy (Sequence[float]): Mixing center for 4 images,
                (x, y).
            img_shape_wh (Sequence[int]): Width and height of sub-image

        Returns:
            tuple[tuple[float]]: Corresponding coordinate of pasting and
                cropping
                - paste_coord (tuple): paste corner coordinate in mosaic image.
                - crop_coord (tuple): crop corner coordinate in mosaic image.
        """
        assert loc in ('top_left', 'top_right', 'bottom_left', 'bottom_right')
        if loc == 'top_left':
            # index0 to top left part of image
            x1, y1, x2, y2 = max(center_position_xy[0] - img_shape_wh[0], 0),                              max(center_position_xy[1] - img_shape_wh[1], 0),                              center_position_xy[0],                              center_position_xy[1]
            crop_coord = img_shape_wh[0] - (x2 - x1), img_shape_wh[1] - (
                y2 - y1), img_shape_wh[0], img_shape_wh[1]

        elif loc == 'top_right':
            # index1 to top right part of image
            x1, y1, x2, y2 = center_position_xy[0],                              max(center_position_xy[1] - img_shape_wh[1], 0),                              min(center_position_xy[0] + img_shape_wh[0],
                                 self.img_scale[0] * 2), \
                             center_position_xy[1]
            crop_coord = 0, img_shape_wh[1] - (y2 - y1), min(
                img_shape_wh[0], x2 - x1), img_shape_wh[1]

        elif loc == 'bottom_left':
            # index2 to bottom left part of image
            x1, y1, x2, y2 = max(center_position_xy[0] - img_shape_wh[0], 0),                              center_position_xy[1],                              center_position_xy[0],                              min(self.img_scale[1] * 2, center_position_xy[1] +
                                 img_shape_wh[1])
            crop_coord = img_shape_wh[0] - (x2 - x1), 0, img_shape_wh[0], min(
                y2 - y1, img_shape_wh[1])

        else:
            # index3 to bottom right part of image
            x1, y1, x2, y2 = center_position_xy[0],                              center_position_xy[1],                              min(center_position_xy[0] + img_shape_wh[0],
                                 self.img_scale[0] * 2), \
                             min(self.img_scale[1] * 2, center_position_xy[1] +
                                 img_shape_wh[1])
            crop_coord = 0, 0, min(img_shape_wh[0],
                                   x2 - x1), min(y2 - y1, img_shape_wh[1])

        paste_coord = x1, y1, x2, y2
        return paste_coord, crop_coord
    def __call__(self, results: dict) -> dict:
        """Data augmentation function.

        The transform steps are as follows:
        1. Randomly generate index list of other images.
        2. Before Mosaic or MixUp need to go through the necessary
            pre_transform, such as MixUp' pre_transform pipeline
            include: 'LoadImageFromFile','LoadAnnotations',
            'Mosaic' and 'RandomAffine'.
        3. Use mix_img_transform function to implement specific
            mix operations.

        Args:
            results (dict): Result dict.

        Returns:
            results (dict): Updated result dict.
        """

        if random.uniform(0, 1) > self.prob:
            return results

        if self.use_cached:
            # Be careful: deep copying can be very time-consuming
            # if results includes dataset.
            dataset = results.pop('dataset', None)
            self.results_cache.append(copy.deepcopy(results))
            if len(self.results_cache) > self.max_cached_images:
                if self.random_pop:
                    index = random.randint(0, len(self.results_cache) - 1)
                else:
                    index = 0
                self.results_cache.pop(index)

            if len(self.results_cache) <= 4:
                return results
        else:
            assert 'dataset' in results
            # Be careful: deep copying can be very time-consuming
            # if results includes dataset.
            dataset = results.pop('dataset', None)

        for _ in range(self.max_refetch):
            # get index of one or three other images
            if self.use_cached:
                indexes = self.get_indexes(self.results_cache)
            else:
                indexes = self.get_indexes(dataset)

            if not isinstance(indexes, collections.abc.Sequence):
                indexes = [indexes]

            if self.use_cached:
                mix_results = [
                    copy.deepcopy(self.results_cache[i]) for i in indexes
                ]
            else:
                # get images information will be used for Mosaic or MixUp
                mix_results = [
                    copy.deepcopy(dataset.get_data_info(index))
                    for index in indexes
                ]

            if self.pre_transform is not None:
                for i, data in enumerate(mix_results):
                    # pre_transform may also require dataset
                    data.update({'dataset': dataset})
                    # before Mosaic or MixUp need to go through
                    # the necessary pre_transform
                    _results = self.pre_transform(data)
                    _results.pop('dataset')
                    mix_results[i] = _results

            if None not in mix_results:
                results['mix_results'] = mix_results
                break
            print('Repeated calculation')
        else:
            raise RuntimeError(
                'The loading pipeline of the original dataset'
                ' always return None. Please check the correctness '
                'of the dataset and its pipeline.')

        # Mosaic or MixUp
        results = self.mix_img_transform(results)

        if 'mix_results' in results:
            results.pop('mix_results')
        results['dataset'] = dataset

        return results


# In[ ]:





# In[37]:


class YOLOv5MixUp:
    """MixUp data augmentation for YOLOv5.

    .. code:: text

    The mixup transform steps are as follows:

        1. Another random image is picked by dataset.
        2. Randomly obtain the fusion ratio from the beta distribution,
            then fuse the target
        of the original image and mixup image through this ratio.

    Required Keys:

    - img
    - gt_bboxes (BaseBoxes[torch.float32]) (optional)
    - gt_bboxes_labels (np.int64) (optional)
    - gt_ignore_flags (bool) (optional)
    - mix_results (List[dict])


    Modified Keys:

    - img
    - img_shape
    - gt_bboxes (optional)
    - gt_bboxes_labels (optional)
    - gt_ignore_flags (optional)


    Args:
        alpha (float): parameter of beta distribution to get mixup ratio.
            Defaults to 32.
        beta (float):  parameter of beta distribution to get mixup ratio.
            Defaults to 32.
        pre_transform (Sequence[dict]): Sequence of transform object or
            config dict to be composed.
        prob (float): Probability of applying this transformation.
            Defaults to 1.0.
        use_cached (bool): Whether to use cache. Defaults to False.
        max_cached_images (int): The maximum length of the cache. The larger
            the cache, the stronger the randomness of this transform. As a
            rule of thumb, providing 10 caches for each image suffices for
            randomness. Defaults to 20.
        random_pop (bool): Whether to randomly pop a result from the cache
            when the cache is full. If set to False, use FIFO popping method.
            Defaults to True.
        max_refetch (int): The maximum number of iterations. If the number of
            iterations is greater than `max_refetch`, but gt_bbox is still
            empty, then the iteration is terminated. Defaults to 15.
    """

    def __init__(self,
                 alpha: float = 32.0,
                 beta: float = 32.0,
                 pre_transform: Sequence[dict] = None,
                 prob: float = 1.0,
                 use_cached: bool = False,
                 max_cached_images: int = 20,
                 random_pop: bool = True,
                 max_refetch: int = 15):
        if use_cached:
            assert max_cached_images >= 2, 'The length of cache must >= 2, '                                            f'but got {max_cached_images}.'
#         super().__init__(
#             pre_transform=pre_transform,
#             prob=prob,
#             use_cached=use_cached,
#             max_cached_images=max_cached_images,
#             random_pop=random_pop,
#             max_refetch=max_refetch)
        self.max_refetch = max_refetch
        self.prob = prob

        self.use_cached = use_cached
        self.max_cached_images = max_cached_images
        self.random_pop = random_pop
        self.results_cache = []

        if pre_transform is None:
            self.pre_transform = None
        else:
            self.pre_transform = transform.Compose(pre_transform)
        self.alpha = alpha
        self.beta = beta

    def get_indexes(self, dataset: Union) -> int:
        """Call function to collect indexes.

        Args:
            dataset (:obj:`Dataset` or list): The dataset or cached list.

        Returns:
            int: indexes.
        """
        return random.randint(0, len(dataset))

    def __call__(self, results: dict) -> dict:
        """YOLOv5 MixUp transform function.

        Args:
            results (dict): Result dict

        Returns:
            results (dict): Updated result dict.
        """
        assert 'mix_results' in results

        retrieve_results = results['mix_results'][0]
        retrieve_img = retrieve_results['img']
        ori_img = results['img']
        assert ori_img.shape == retrieve_img.shape

        # Randomly obtain the fusion ratio from the beta distribution,
        # which is around 0.5
        ratio = np.random.beta(self.alpha, self.beta)
        mixup_img = (ori_img * ratio + retrieve_img * (1 - ratio))

        retrieve_gt_bboxes = retrieve_results['gt_bboxes']
        retrieve_gt_bboxes_labels = retrieve_results['gt_bboxes_labels']
        retrieve_gt_ignore_flags = retrieve_results['gt_ignore_flags']

        mixup_gt_bboxes = retrieve_gt_bboxes.cat(
            (results['gt_bboxes'], retrieve_gt_bboxes), dim=0)
        mixup_gt_bboxes_labels = np.concatenate(
            (results['gt_bboxes_labels'], retrieve_gt_bboxes_labels), axis=0)
        mixup_gt_ignore_flags = np.concatenate(
            (results['gt_ignore_flags'], retrieve_gt_ignore_flags), axis=0)
        if 'gt_masks' in results:
            assert 'gt_masks' in retrieve_results
#             mixup_gt_masks = results['gt_masks'].cat(
#                 [results['gt_masks'], retrieve_results['gt_masks']])
            raise TypeError("unsuuported type mask in MixUp")
            results['gt_masks'] = mixup_gt_masks

        results['img'] = mixup_img.astype(np.uint8)
        results['img_shape'] = mixup_img.shape
        results['gt_bboxes'] = mixup_gt_bboxes
        results['gt_bboxes_labels'] = mixup_gt_bboxes_labels
        results['gt_ignore_flags'] = mixup_gt_ignore_flags

        return results


# In[38]:


# DataPrecessor only neccessary(used)
class YOLOv5DetDataPreprocessor(Module):
    def __init__(self, data_root, file, prefix):
        Dataloder(root = data_root + file,
                   transform = transform.ImageNormalize(mean=[0., 0., 0.],
                                                       std=[255., 255., 255.]))
        #bgr2rgb has been write during dataload


# In[39]:


#module
class detector(Module):
    def __init__(self):
        super(detector, self).__init__()
        pass
#     def data_preprocessor(self,input):
#         mean=[0., 0., 0.]
#         std=[255., 255., 255.]
#         bgr_to_rgb=True
#         return 
    def excute(self):
        pass
        
    
# model = dict(
#     type='YOLODetector',
#     data_preprocessor=dict(
#         type='YOLOv5DetDataPreprocessor',
#         mean=[0., 0., 0.],
#         std=[255., 255., 255.],
#         bgr_to_rgb=True),
#     backbone=dict(
#         type='RemNet',
#         arch='P5',
#         last_stage_out_channels=last_stage_out_channels,
#         deepen_factor=deepen_factor,
#         widen_factor=widen_factor,
#         norm_cfg=norm_cfg,
#         act_cfg=dict(type='SiLU', inplace=True)),
#     neck=dict(
#         type='RemDetPAFPN', # YOLOv8PAFPN  RemDetPAFPN
#         deepen_factor=deepen_factor,
#         widen_factor=widen_factor,
#         in_channels=[256, 512, last_stage_out_channels],
#         out_channels=[256, 512, last_stage_out_channels],
#         num_csp_blocks=3,
#         norm_cfg=norm_cfg,
#         act_cfg=dict(type='SiLU', inplace=True)),
#     bbox_head=dict(
#         type='YOLOv8Head',
#         head_module=dict(
#             type='YOLOv8HeadModule',
#             num_classes=num_classes,
#             in_channels=[256, 512, last_stage_out_channels],
#             widen_factor=widen_factor,
#             reg_max=16,
#             norm_cfg=norm_cfg,
#             act_cfg=dict(type='SiLU', inplace=True),
#             featmap_strides=strides),
#         prior_generator=dict(
#             type='mmdet.MlvlPointGenerator', offset=0.5, strides=strides),
#         bbox_coder=dict(type='YOLODistancePointBBoxCoder'),  # TODO need modify
#         # scaled based on number of detection layers
#         loss_cls=dict(
#             type='mmdet.CrossEntropyLoss',
#             use_sigmoid=True,
#             reduction='none',
#             loss_weight=loss_cls_weight),
#         loss_bbox=dict(
#             type='YOLO_IoULoss',
#             iou_mode='ciou',
#             bbox_format='xyxy',
#             reduction='sum',
#             loss_weight=loss_bbox_weight,
#             return_iou=False),
#         loss_dfl=dict(
#             type='mmdet.DistributionFocalLoss',
#             reduction='mean',
#             loss_weight=loss_dfl_weight)),
#     train_cfg=dict(
#         assigner=dict(
#             type='BatchTaskAlignedAssigner',
#             num_classes=num_classes,
#             use_ciou=True,
#             topk=tal_topk,
#             alpha=tal_alpha,
#             beta=tal_beta,
#             eps=1e-9)),
#     test_cfg=model_test_cfg)

# model = dict(
#     backbone=dict(
#         last_stage_out_channels=last_stage_out_channels,
#         deepen_factor=deepen_factor,
#         widen_factor=widen_factor),
#     neck=dict(
#         deepen_factor=deepen_factor,
#         widen_factor=widen_factor,
#         in_channels=[256, 512, last_stage_out_channels],
#         out_channels=[256, 512, last_stage_out_channels]),
#     bbox_head=dict(
#         head_module=dict(
#             widen_factor=widen_factor,
#             in_channels=[256, 512, last_stage_out_channels])))


# In[40]:


class SiLU(Module):
    def __init__(self):
        super().__init__()
    def execute(self, x):
        return x * x.sigmoid()
# class ReLU(Module):
#     def __init__(self):
#         super().__init__()
#     def execute(self, x):
#         return jt.maximum(0, x)


# In[41]:


class ConvModule(Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, Tuple[int, int]],
                 stride: Union[int, Tuple[int, int]] = 1,
                 padding: Union[int, Tuple[int, int]] = 0,
                 dilation: Union[int, Tuple[int, int]] = 1,
                 groups: int = 1,
                 bias: Union[bool, str] = 'auto',
                 conv_cfg: Optional[Dict] = None,
                 norm_cfg: Optional[Dict] = None,
                 act_cfg: Optional[Dict] = dict(type='ReLU'),
                 inplace: bool = True,
                 with_spectral_norm: bool = False,
                 padding_mode: str = 'zeros',
                 order: tuple = ('conv', 'norm', 'act'),
                 efficient_conv_bn_eval: bool = False):
        super().__init__()
        assert conv_cfg is None or isinstance(conv_cfg, dict)
        assert norm_cfg is None or isinstance(norm_cfg, dict)
        assert act_cfg is None or isinstance(act_cfg, dict)
        official_padding_mode = ['zeros', 'circular']
        # TODO change here if need
        assert padding_mode == 'zeros','I do not know how to use nn.pad in jittor and the modle only use default mode'
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.inplace = inplace
        self.with_spectral_norm = with_spectral_norm
        self.with_explicit_padding = padding_mode not in official_padding_mode
        self.order = order
        assert isinstance(self.order, tuple) and len(self.order) == 3
        assert set(order) == {'conv', 'norm', 'act'}

        self.with_norm = norm_cfg is not None
        self.with_activation = act_cfg is not None
        # if the conv layer is before a norm layer, bias is unnecessary.
        if bias == 'auto':
            bias = not self.with_norm
        self.with_bias = bias
        
        conv_padding = padding
        self.conv = nn.Conv(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=conv_padding,
            dilation=dilation,
            groups=groups,
            bias=bias
        )
        self.in_channels = self.conv.in_channels
        self.out_channels = self.conv.out_channels
        self.kernel_size = self.conv.kernel_size
        self.stride = self.conv.stride
        self.padding = padding
        self.dilation = self.conv.dilation
        self.transposed = self.conv.transposed
        self.output_padding = self.conv.output_padding
        self.groups = self.conv.groups
        
        #if jittor support spectral norm,change here TODO
        if self.with_spectral_norm:
            raise TypeError('if you need, please change here')
#             self.conv = nn.utils.spectral_norm(self.conv)
        if self.with_norm:
            assert self.norm_cfg['type'] == 'BN','only support bn,if want else, please add'
            # norm layer is after conv layer
            if order.index('norm') > order.index('conv'):
                norm_channels = out_channels
            else:
                norm_channels = in_channels
                self.norm_name = nn.BatchNorm(
                    num_features = norm_channels
                )   
        else: self.norm_name = None
        assert efficient_conv_bn_eval == False,'the efficient_conv_bn_eval not used, if want please write'
#         self.turn_on_efficient_conv_bn_eval(efficient_conv_bn_eval)
        
        if self.with_activation:
            assert self.act_cfg['type'] in ['SiLU', 'ReLU', 'LeakyReLU']
            if self.act_cfg['type'] == 'SiLU':
                self.activate = SiLU()
            elif self.act_cfg['type'] == 'ReLU':
                self.activate = nn.ReLU()
            else: self.activate = nn.LeakyReLU()
        # Use msra init by default
        self.init_weights()
    def init_weights(self):
        # 1. It is mainly for customized conv layers with their own
        #    initialization manners by calling their own ``init_weights()``,
        #    and we do not want ConvModule to override the initialization.
        # 2. For customized conv layers without their own initialization
        #    manners (that is, they don't have their own ``init_weights()``)
        #    and PyTorch's conv layers, they will be initialized by
        #    this method with default ``kaiming_init``.
        # Note: For PyTorch's conv layers, they will be overwritten by our
        #    initialization implementation using default ``kaiming_init``.
        if not hasattr(self.conv, 'init_weights'):
            if self.with_activation and self.act_cfg['type'] == 'LeakyReLU':
                nonlinearity = 'leaky_relu'
                a = self.act_cfg.get('negative_slope', 0.01)
            else:
                nonlinearity = 'relu'
                a = 0
            kaiming_norm_(self.conv, a=a, nonlinearity=nonlinearity)
            # Notcie that I do not search in mmengine for defination for kaiming_init : TODO
        if self.with_norm:
            constant_(self.norm, 1, bias=0)
    def execute(self, x):
        layer_index = 0
        while layer_index < len(self.order):
            layer = self.order[layer_index]
            if layer == 'conv':
#                 if self.with_explicit_padding:
#                     x = self.padding_layer(x)
                # if the next operation is norm and we have a norm layer in
                # eval mode and we have enabled `efficient_conv_bn_eval` for
                # the conv operator, then activate the optimized forward and
                # skip the next norm operator since it has been fused
#                 if layer_index + 1 < len(self.order) and \
#                         self.order[layer_index + 1] == 'norm' and \
#                         self.with_norm and not self.norm.training and \
#                         self.efficient_conv_bn_eval_forward is not None:
#                     self.conv.forward = partial(
#                         self.efficient_conv_bn_eval_forward, self.norm,
#                         self.conv)
#                     layer_index += 1
#                     x = self.conv(x)
#                     del self.conv.forward
#                 else:
                x = self.conv(x)
            elif layer == 'norm' and self.with_norm:
                x = self.norm(x)
            elif layer == 'act' and self.with_activation:
                x = self.activate(x)
            layer_index += 1
        return x


# In[ ]:





# In[42]:


class SPPFBottleneck(Module):
    """Spatial pyramid pooling - Fast (SPPF) layer for
    YOLOv5, YOLOX and PPYOLOE by Glenn Jocher

    Args:
        in_channels (int): The input channels of this Module.
        out_channels (int): The output channels of this Module.
        kernel_sizes (int, tuple[int]): Sequential or number of kernel
            sizes of pooling layers. Defaults to 5.
        use_conv_first (bool): Whether to use conv before pooling layer.
            In YOLOv5 and YOLOX, the para set to True.
            In PPYOLOE, the para set to False.
            Defaults to True.
        mid_channels_scale (float): Channel multiplier, multiply in_channels
            by this amount to get mid_channels. This parameter is valid only
            when use_conv_fist=True.Defaults to 0.5.
        conv_cfg (dict): Config dict for convolution layer. Defaults to None.
            which means using conv2d. Defaults to None.
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to dict(type='BN', momentum=0.03, eps=0.001).
        act_cfg (dict): Config dict for activation layer.
            Defaults to dict(type='SiLU', inplace=True).
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__( self,
                 in_channels: int,
                 out_channels: int,
                 kernel_sizes: Union[int, Sequence[int]] = 5,
                 use_conv_first: bool = True,
                 mid_channels_scale: float = 0.5,
                 conv_cfg=None,
                 norm_cfg=dict(
                     type='BN', momentum=0.03, eps=0.001),
                 act_cfg= dict(type='SiLU', inplace=True),
                 init_cfg= None):
        super().__init__()

        if use_conv_first:
            mid_channels = int(in_channels * mid_channels_scale)
            self.conv1 = ConvModule(
                in_channels,
                mid_channels,
                1,
                stride=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
        else:
            mid_channels = in_channels
            self.conv1 = None
        self.kernel_sizes = kernel_sizes
        if isinstance(kernel_sizes, int):
            self.poolings = nn.MaxPool2d(
                kernel_size=kernel_sizes, stride=1, padding=kernel_sizes // 2)
            conv2_in_channels = mid_channels * 4
        else:
            self.poolings = nn.ModuleList([
                nn.MaxPool2d(kernel_size=ks, stride=1, padding=ks // 2)
                for ks in kernel_sizes
            ])
            conv2_in_channels = mid_channels * (len(kernel_sizes) + 1)

        self.conv2 = ConvModule(
            conv2_in_channels,
            out_channels,
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

    def execute(self, x):
        """Forward process
        Args:
            x (Tensor): The input tensor.
        """
        if self.conv1:
            x = self.conv1(x)
        if isinstance(self.kernel_sizes, int):
            y1 = self.poolings(x)
            y2 = self.poolings(y1)
            x = jt.cat([x, y1, y2, self.poolings(y2)], dim=1)
        else:
            x = jt.cat(
                [x] + [pooling(x) for pooling in self.poolings], dim=1)
        x = self.conv2(x)
        return x


# In[ ]:





# In[43]:


# Notice : only support RemDet
class RepDWConv(Module):
    """
        remove identity
    """

    def __init__(self,
                 in_channels: int,  # c1
                 out_channels: int,  # c2
                 kernel_size: Union[int, Tuple[int]] = 3,
                 stride: Union[int, Tuple[int]] = 1,
                 padding: Union[int, Tuple[int]] = 1,
                 dilation: Union[int, Tuple[int]] = 1,
                 groups: Optional[int] = 1,
                 padding_mode: Optional[str] = 'zeros',
                 norm_cfg= dict(
                     type='BN', momentum=0.03, eps=0.001),
                 act_cfg= dict(type='ReLU', inplace=True), 
                 use_se: bool = False,
                 use_alpha: bool = False,
                 use_bn_first=True,
                 deploy: bool = False):
        super().__init__()
        self.deploy = deploy
        self.groups = math.gcd(in_channels, out_channels)
        self.in_channels = in_channels
        self.out_channels = out_channels

        assert kernel_size == 3
        assert padding == 1

        padding_11 = padding - kernel_size // 2

#         self.nonlinearity = MODELS.build(act_cfg)
#         assert act_cfg['type'] == 'ReLU','module only use the activate_fun if need please add'
        activation = ['SiLU','ReLU']
        assert act_cfg['type'] in activation,'only support these activation'
        if act_cfg['type'] == 'ReLU':
            self.nonlinearity = nn.ReLU()
        else: self.nonlinearity = SiLU()

        if use_se:
            raise NotImplementedError('se block not supported yet')
        else:
            self.se = nn.Identity()

        if use_alpha:
            alpha = jt.ones([
                1,
            ], dtype=jt.float32, need_grad=True)
#             self.alpha = nn.Parameter(alpha, need_grad=True)
        else:
            self.alpha = None

        if deploy:
            self.rbr_reparam = nn.Conv(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=self.groups,
                bias=True,
                )

        else:
            if use_bn_first and (out_channels == in_channels) and stride == 1:
#                 self.rbr_identity = build_norm_layer(
#                     norm_cfg, num_features=in_channels)[1]
                assert norm_cfg['type'] == 'BN','if needed,please change'
                self.rbr_identity = nn.BatchNorm(momentum=0.03, eps=0.001,num_features=in_channels)
            else:
                self.rbr_identity = None

            self.rbr_dense = ConvModule(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=self.groups,
                bias=False,
                norm_cfg=norm_cfg,
                act_cfg=None)
            self.rbr_1x1 = ConvModule(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=stride,
                padding=padding_11,
                bias=False,
                norm_cfg=norm_cfg,
                act_cfg=None)

    def execute(self, inputs) -> jt.Var:
        """Forward process.
        Args:
            inputs (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor.
        """
        if hasattr(self, 'rbr_reparam'):
            return self.nonlinearity(self.se(self.rbr_reparam(inputs)))

        inputs = self.rbr_identity(inputs)
        if self.alpha:
            return self.nonlinearity(
                self.se(
                    self.rbr_dense(inputs) +
                    self.alpha * self.rbr_1x1(inputs)))
        else:
            return self.nonlinearity(
                self.se(
                    self.rbr_dense(inputs) + self.rbr_1x1(inputs)))

    def get_equivalent_kernel_bias(self):
        """Derives the equivalent kernel and bias in a differentiable way.

        Returns:
            tuple: Equivalent kernel and bias
        """
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        if self.alpha:
            return kernel3x3 + self.alpha * self._pad_1x1_to_3x3_tensor(
                kernel1x1) + kernelid, bias3x3 + self.alpha * bias1x1 + biasid
        else:
            return kernel3x3 + self._pad_1x1_to_3x3_tensor(
                kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        """Pad 1x1 tensor to 3x3.
        Args:
            kernel1x1 (Tensor): The input 1x1 kernel need to be padded.

        Returns:
            Tensor: 3x3 kernel after padded.
        """
        if kernel1x1 is None:
            return 0
        else:
            return jt.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch: nn.Module) -> Tuple[np.ndarray, jt.Var]:
        """Derives the equivalent kernel and bias of a specific branch layer.

        Args:
            branch (nn.Module): The layer that needs to be equivalently
                transformed, which can be nn.Sequential or nn.Batchnorm2d

        Returns:
            tuple: Equivalent kernel and bias
        """
        if branch is None:
            return 0, 0
        if isinstance(branch, ConvModule):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            assert isinstance(branch, (nn.SyncBatchNorm, nn.BatchNorm2d))
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros((self.in_channels, input_dim, 3, 3),
                                        dtype=np.float32)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = jt.from_numpy(kernel_value).to(
                    branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def switch_to_deploy(self):
        """Switch to deploy mode."""
        if hasattr(self, 'rbr_reparam'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.rbr_reparam = Conv2d(
            in_channels=self.rbr_dense.conv.in_channels,
            out_channels=self.rbr_dense.conv.out_channels,
            kernel_size=self.rbr_dense.conv.kernel_size,
            stride=self.rbr_dense.conv.stride,
            padding=self.rbr_dense.conv.padding,
            dilation=self.rbr_dense.conv.dilation,
            groups=self.rbr_dense.conv.groups,
            bias=True)
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('rbr_dense')
        self.__delattr__('rbr_1x1')
        if hasattr(self, 'rbr_identity'):
            self.__delattr__('rbr_identity')
        if hasattr(self, 'id_tensor'):
            self.__delattr__('id_tensor')
        self.deploy = True


# In[ ]:





# In[44]:


class DarknetBottleneck(Module):
    """The basic bottleneck block used in Darknet.

    Each ResBlock consists of two ConvModules and the input is added to the
    final output. Each ConvModule is composed of Conv, BN, and LeakyReLU.
    The first convLayer has filter size of 1x1 and the second one has the
    filter size of 3x3.

    Args:
        in_channels (int): The input channels of this Module.
        out_channels (int): The output channels of this Module.
        expansion (float): The kernel size of the convolution.
            Defaults to 0.5.
        add_identity (bool): Whether to add identity to the out.
            Defaults to True.
        use_depthwise (bool): Whether to use depthwise separable convolution.
            Defaults to False.
        conv_cfg (dict): Config dict for convolution layer. Defaults to None,
            which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to dict(type='BN').
        act_cfg (dict): Config dict for activation layer.
            Defaults to dict(type='Swish').
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 expansion: float = 0.5,
                 add_identity: bool = True,
                 use_depthwise: bool = False,
                 conv_cfg= None,
                 norm_cfg= dict(
                     type='BN', momentum=0.03, eps=0.001),
                 act_cfg= dict(type='SiLU'),
                 init_cfg= None) -> None:
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        assert not use_depthwise,'not support'
        conv = DepthwiseSeparableConvModule if use_depthwise else ConvModule
        self.conv1 = ConvModule(
            in_channels,
            hidden_channels,
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.conv2 = conv(
            hidden_channels,
            out_channels,
            3,
            stride=1,
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.add_identity =             add_identity and in_channels == out_channels

    def execute(self, x: jt.Var) -> jt.Var:
        """Forward function."""
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)

        if self.add_identity:
            return out + identity
        else:
            return out


# In[ ]:





# In[45]:


class ChannelC2f(Module):

    def __init__(
            self,
            c1: int,
            c2: int,
            e: float = 1,
            n: int = 1,
            shortcut: bool = True,  # shortcut
            conv_cfg= None,
            norm_cfg= dict(type='BN', momentum=0.03, eps=0.001),
            act_cfg= dict(type='SiLU', inplace=True),
            init_cfg = None) -> None:
        super().__init__()

        self.c = int(c2 * e)
        self.cv1 = ConvModule(
            c1,
            2 * self.c,
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.cv2 = ConvModule(
            (2 + n) * self.c,
            c2,
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.m = nn.ModuleList(
            DarknetBottleneck(
                self.c,
                self.c,
                expansion=0.25,
                kernel_size=(3, 3),
                padding=(1, 1),
                add_identity=shortcut,
                use_depthwise=False,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg) for _ in range(n))

    def execute(self, x: jt.Var) -> jt.Var:
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(jt.cat(y, 1))


# In[ ]:





# In[46]:


class GatedFFN(Module):  # 0608
    def __init__(self, c1: int,
                 c2: int,
                 n: int = 1,
                 shortcut: bool = False,
                 g: int = 1,
                 e: int = 3,
                 conv_cfg = None,
                 norm_cfg = dict(type='BN', momentum=0.03, eps=0.001),
                 act_cfg = dict(type='SiLU', inplace=True)):
        super().__init__()
        self.n = n
        self.c = int(c2 * e)
        self.proj = ConvModule(c1, 2 * self.c,
                               kernel_size=1, stride=1, padding=0,
                               conv_cfg=conv_cfg,
                               norm_cfg=norm_cfg,
                               act_cfg=act_cfg)
        self.rep = RepDWConv(self.c, self.c)  # deploy=True
        self.m = nn.ModuleList(
            ConvModule(self.c, self.c, kernel_size=3, stride=1, padding=autopad(3), groups=self.c,
                       conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=None) for _ in range(n - 1)
        )
        self.act = nn.GELU()
        self.cv2 = ConvModule(self.c, c2,
                              kernel_size=1, stride=1, padding=0,
                              conv_cfg=conv_cfg,
                              norm_cfg=norm_cfg,
                              act_cfg=None)
        self.add = shortcut and c1 == c2

    def execute(self, x):
        shortcut = x.clone()
        x, z = self.proj(x).split([self.c, self.c], 1)
        x = self.rep(x)
        if self.n != 1:
            for m in self.m:
                x = m(x)
        x = x * self.act(z)
        x = self.cv2(x)
        return x + shortcut if self.add else x


# In[ ]:





# In[47]:


class CED(Module):
    def __init__(self, c1, c2, e=0.5,
                 norm_cfg = dict(type='BN', momentum=0.03, eps=0.001),
                 act_cfg = dict(type='SiLU', inplace=True)):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = ConvModule(c1, self.c, kernel_size=1,
                              stride=1, padding=0,
                              norm_cfg=norm_cfg,
                              act_cfg=act_cfg)
        self.cv2 = ConvModule(self.c * 4, c2, kernel_size=1,
                              stride=1, padding=0,
                              norm_cfg=norm_cfg,
                              act_cfg=None)

        # self.dwconv = nn.Sequential(RepDWConv(self.c, self.c), nn.SiLU())
        self.dwconv = ConvModule(self.c, self.c,
                                 kernel_size=3, stride=1, padding=autopad(3),
                                 groups=self.c,
                                 norm_cfg=norm_cfg,
                                 act_cfg=act_cfg)

    def execute(self, x):
        x = self.dwconv(self.cv1(x))
        x = jt.cat([
            x[..., ::2, ::2],
            x[..., 1::2, ::2],
            x[..., ::2, 1::2],
            x[..., 1::2, 1::2],
        ], dim=1)
        x = self.cv2(x)
        return x


# In[ ]:





# In[48]:


class RemDetPAFPN(Module):
    """Path Aggregation Network used in YOLOv8.

    Args:
        in_channels (List[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale)
        deepen_factor (float): Depth multiplier, multiply number of
            blocks in CSP layer by this amount. Defaults to 1.0.
        widen_factor (float): Width multiplier, multiply number of
            channels in each layer by this amount. Defaults to 1.0.
        num_csp_blocks (int): Number of bottlenecks in CSPLayer. Defaults to 1.
        freeze_all(bool): Whether to freeze the model
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to dict(type='BN', momentum=0.03, eps=0.001).
        act_cfg (dict): Config dict for activation layer.
            Defaults to dict(type='SiLU', inplace=True).
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 in_channels: List[int],
                 out_channels: Union[List[int], int],
                 deepen_factor: float = 1.0,
                 widen_factor: float = 1.0,
                 num_csp_blocks: int = 3,
                 freeze_all: bool = False,
                 norm_cfg = dict(
                     type='BN', momentum=0.03, eps=0.001),
                 act_cfg = dict(type='SiLU', inplace=True),
                 init_cfg = None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.deepen_factor = deepen_factor
        self.widen_factor = widen_factor
        self.upsample_feats_cat_first = upsample_feats_cat_first
        self.freeze_all = freeze_all
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg

        self.reduce_layers = nn.ModuleList()
        for idx in range(len(in_channels)):
            self.reduce_layers.append(self.build_reduce_layer(idx))

        # build top-down blocks
        self.upsample_layers = nn.ModuleList()
        self.top_down_layers = nn.ModuleList()
        for idx in range(len(in_channels) - 1, 0, -1):
            self.upsample_layers.append(self.build_upsample_layer(idx))
            self.top_down_layers.append(self.build_top_down_layer(idx))

        # build bottom-up blocks
        self.downsample_layers = nn.ModuleList()
        self.bottom_up_layers = nn.ModuleList()
        for idx in range(len(in_channels) - 1):
            self.downsample_layers.append(self.build_downsample_layer(idx))
            self.bottom_up_layers.append(self.build_bottom_up_layer(idx))

        self.out_layers = nn.ModuleList()
        for idx in range(len(in_channels)):
            self.out_layers.append(self.build_out_layer(idx))

    def _freeze_all(self):
        """Freeze the model."""
        for m in self.modules():
            if isinstance(m, _BatchNorm):
                m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def train(self, mode=True):
        """Convert the model into training mode while keep the normalization
        layer freezed."""
        if mode:
            super().train()
        if self.freeze_all:
            self._freeze_all()

    def execute(self, inputs: List[jt.Var]) -> tuple:
        """Forward function."""
        assert len(inputs) == len(self.in_channels)
        # reduce layers
        reduce_outs = []
        for idx in range(len(self.in_channels)):
            reduce_outs.append(self.reduce_layers[idx](inputs[idx]))

        # top-down path
        inner_outs = [reduce_outs[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_high = inner_outs[0]
            feat_low = reduce_outs[idx - 1]
            upsample_feat = self.upsample_layers[len(self.in_channels) - 1 -
                                                 idx](
                                                     feat_high)
            if self.upsample_feats_cat_first:
                top_down_layer_inputs = jt.cat([upsample_feat, feat_low], 1)
            else:
                top_down_layer_inputs = jt.cat([feat_low, upsample_feat], 1)
            inner_out = self.top_down_layers[len(self.in_channels) - 1 - idx](
                top_down_layer_inputs)
            inner_outs.insert(0, inner_out)

        # bottom-up path
        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            feat_high = inner_outs[idx + 1]
            downsample_feat = self.downsample_layers[idx](feat_low)
            out = self.bottom_up_layers[idx](
                jt.cat([downsample_feat, feat_high], 1))
            outs.append(out)

        # out_layers
        results = []
        for idx in range(len(self.in_channels)):
            results.append(self.out_layers[idx](outs[idx]))

        return tuple(results)
    def build_reduce_layer(self, idx: int) -> nn.Module:
        """build reduce layer.

        Args:
            idx (int): layer idx.

        Returns:
            nn.Module: The reduce layer.
        """
        return nn.Identity()

    def build_top_down_layer(self, idx: int) -> nn.Module:
        """build top down layer.

        Args:
            idx (int): layer idx.

        Returns:
            nn.Module: The top down layer.
        """
        return ChannelC2f(
            make_divisible((self.in_channels[idx - 1] + self.in_channels[idx]),
                           self.widen_factor),
            make_divisible(self.out_channels[idx - 1], self.widen_factor),
            n=make_round(self.num_csp_blocks, self.deepen_factor),
            shortcut=False,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

    def build_bottom_up_layer(self, idx: int) -> nn.Module:
        """build bottom up layer.

        Args:
            idx (int): layer idx.

        Returns:
            nn.Module: The bottom up layer.
        """
        return ChannelC2f(
            make_divisible(
                (self.out_channels[idx] + self.out_channels[idx + 1]),
                self.widen_factor),
            make_divisible(self.out_channels[idx + 1], self.widen_factor),
            n=make_round(self.num_csp_blocks, self.deepen_factor),
            shortcut=False,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
    
    def build_upsample_layer(self, *args, **kwargs) -> nn.Module:
        """build upsample layer."""
        return nn.Upsample(scale_factor=2, mode='nearest')
    
    def build_downsample_layer(self, idx: int) -> nn.Module:
        """build downsample layer.

        Args:
            idx (int): layer idx.

        Returns:
            nn.Module: The downsample layer.
        """
        return ConvModule(
            make_divisible(self.in_channels[idx], self.widen_factor),
            make_divisible(self.in_channels[idx], self.widen_factor),
            kernel_size=3,
            stride=2,
            padding=1,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
    
    def build_out_layer(self, *args, **kwargs) -> nn.Module:
        """build out layer."""
        return nn.Identity()


# In[ ]:





# In[49]:


class RemNet(Module):
    """
        remdet backbone
    """
    # From left to right:
    # in_channels, out_channels, num_blocks, add_identity, use_spp, is_first
    # the final out_channels will be set according to the param.
    arch_settings = {
        # in_channels, out_channels, num_blocks, add_identity, use_spp, is_first, expansion
        'P5': [[64, 128, 3, True, False, True, 2], [128, 256, 3, True, False, False, 1],
               [256, 512, 6, True, False, False, 1], [512, None, 3, True, True, False, 1]],
    }

    def __init__(self,
                 arch: str = 'P5',
                 last_stage_out_channels: int = 1024,
                 plugins: Union[dict, List[dict]] = None,
                 deepen_factor: float = 1.0,
                 widen_factor: float = 1.0,
                 input_channels: int = 3,
                 out_indices: Tuple[int] = (2, 3, 4),
                 frozen_stages: int = -1,
                 norm_cfg = dict(
                     type='BN', momentum=0.03, eps=0.001),
                 act_cfg = dict(type='SiLU', inplace=True),
                 norm_eval: bool = False,
                 channel_expansion_ratio: int = 1,
                 init_cfg = None):
        self.arch_settings[arch][-1][1] = last_stage_out_channels
        self.channel_expansion_ratio = channel_expansion_ratio
        super().__init__()
        self.num_stages = len(arch_setting)
        self.arch_setting = arch_setting

        assert set(out_indices).issubset(
            i for i in range(len(arch_setting) + 1))

        if frozen_stages not in range(-1, len(arch_setting) + 1):
            raise ValueError('"frozen_stages" must be in range(-1, '
                             'len(arch_setting) + 1). But received '
                             f'{frozen_stages}')

        self.input_channels = input_channels
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.widen_factor = widen_factor
        self.deepen_factor = deepen_factor
        self.norm_eval = norm_eval
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.plugins = plugins

        self.stem = self.build_stem_layer()
        self.layers = ['stem']

        for idx, setting in enumerate(arch_setting):
            stage = []
            stage += self.build_stage_layer(idx, setting)
            if plugins is not None:
                raise TypeError("not support")
#                 stage += self.make_stage_plugins(plugins, idx, setting)
            self.add_module(f'stage{idx + 1}', nn.Sequential(*stage))
            self.layers.append(f'stage{idx + 1}')

    def build_stem_layer(self) -> nn.Module:
        """Build a stem layer."""
        return ConvModule(
            self.input_channels,
            make_divisible(self.arch_setting[0][0], self.widen_factor),
            kernel_size=3,
            stride=2,
            padding=1,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

    def build_stage_layer(self, stage_idx: int, setting: list) -> list:
        """Build a stage layer.

        Args:
            stage_idx (int): The index of a stage layer.
            setting (list): The architecture setting of a stage layer.
        """
        in_channels, out_channels, num_blocks, add_identity, use_spp, is_first, expansion = setting

        in_channels = make_divisible(in_channels, self.widen_factor)
        out_channels = make_divisible(out_channels, self.widen_factor)
        num_blocks = make_round(num_blocks, self.deepen_factor)
        stage = []
        conv_layer = CED(in_channels,
                         out_channels,
                         e=expansion,
                         norm_cfg=self.norm_cfg,
                         act_cfg=self.act_cfg) if not is_first else ConvModule(in_channels,
                                                                               out_channels,
                                                                               kernel_size=3,
                                                                               stride=2,
                                                                               padding=1,
                                                                               norm_cfg=self.norm_cfg,
                                                                               act_cfg=self.act_cfg)
        stage.append(conv_layer)
        csp_layer = GatedFFN(
            out_channels,
            out_channels,
            n=num_blocks,
            shortcut=add_identity,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg
        )
        stage.append(csp_layer)
        if use_spp:
            spp = SPPFBottleneck(
                out_channels,
                out_channels,
                kernel_sizes=5,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)
            stage.append(spp)
        return stage

#     def init_weights(self):
#         """Initialize the parameters."""
#         if self.init_cfg is None:
#             for m in self.modules():
#                 if isinstance(m, jt.nn.Conv2d):
#                     # In order to be consistent with the source code,
#                     # reset the Conv2d initialization parameters
#                     m.reset_parameters()
#         else:
#             super().init_weights()
    def make_stage_plugins(self, plugins, stage_idx, setting):
        pass


    def _freeze_stages(self):
        """Freeze the parameters of the specified stage so that they are no
        longer updated."""
        if self.frozen_stages >= 0:
            for i in range(self.frozen_stages + 1):
                m = getattr(self, self.layers[i])
                m.eval()
                for param in m.parameters():
                    param.need_grad = False

    def train(self, mode: bool = True):
        """Convert the model into training mode while keep normalization layer
        frozen."""
        if mode:
            super().train()
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()

    def execute(self, x: jt.Var) -> tuple:
        """Forward batch_inputs from the data_preprocessor."""
        outs = []
        for i, layer_name in enumerate(self.layers):
            layer = getattr(self, layer_name)
            x = layer(x)
            if i in self.out_indices:
                outs.append(x)

        return tuple(outs)


# In[ ]:





# In[50]:


_backend_args = None
_multiscale_resize_transforms = transform.Compose([YOLOv5KeepRatioResize(scale = (640, 640, )),
                                                  LetterResize(scale = (640, 640, ), 
                                                               allow_scale_up = False, 
                                                               pad_val=dict(img=114)),
                                                  YOLOv5KeepRatioResize(scale = (320, 320, )),
                                                  LetterResize(scale = (320, 320, ), 
                                                               allow_scale_up = False, 
                                                               pad_val=dict(img=114)),
                                                  YOLOv5KeepRatioResize(scale = (960, 960, )),
                                                  LetterResize(scale = (960, 960, ), 
                                                               allow_scale_up = False, 
                                                               pad_val=dict(img=114)),]
                                                 )
album_train_transform = [Blur(p=0.01),
                         MedianBlur(p=0.01),
                         ToGray(p=0.01),
                         CLAHE(p=0.01)]
backend_args = None


switch_pipeline = transform.Compose([
    LoadImageFromFile(backend_args=None),
    LoadYOLOAnnotations(with_bbox = True),
    YOLOv5KeepRatioResize(scale = (640, 640, )),
                                                  LetterResize(scale = (640, 640, ), 
                                                               allow_scale_up = True, 
                                                               pad_val=dict(img=114.0)),
                                                  YOLOv5RandomAffine(border_val=(
                                                                    114,
                                                                    114,
                                                                    114,
                                                                ),
                                                                max_aspect_ratio=100,
                                                                max_rotate_degree=0.0,
                                                                max_shear_degree=0.0,
                                                                scaling_ratio_range=(
                                                                    0.5,
                                                                    1.5,
                                                                ),
                                                                    ),
                                     Albu(bbox_params=dict(
                                            format='pascal_voc',
                                            label_fields=[
                                                'gt_bboxes_labels',
                                                'gt_ignore_flags',
                                            ],
                                            type='BboxParams'),
                                        keymap=dict(gt_bboxes='bboxes', img='image'),
                                        transforms=[
                                            dict(p=0.01, type='Blur'),
                                            dict(p=0.01, type='MedianBlur'),
                                            dict(p=0.01, type='ToGray'),
                                            dict(p=0.01, type='CLAHE'),
                                        ]),
                                     YOLOv5HSVRandomAug,
                                     RandomFlip(prob = 0.5),
                                     PackDetInputs(meta_keys=(
                                        'img_id',
                                        'img_path',
                                        'ori_shape',
                                        'img_shape',
                                        'flip',
                                        'flip_direction',
                                    ))
                                    ])

last_transform = [Albu(bbox_params=dict(
                    format='pascal_voc',
                    label_fields=[
                        'gt_bboxes_labels',
                        'gt_ignore_flags',
                    ],
                    type='BboxParams'),
                keymap=dict(gt_bboxes='bboxes', img='image'),
                transforms=[
                    dict(p=0.01, type='Blur'),
                    dict(p=0.01, type='MedianBlur'),
                    dict(p=0.01, type='ToGray'),
                    dict(p=0.01, type='CLAHE'),
                ]),
                  YOLOv5HSVRandomAug,
                  RandomFlip(prob = 0.5),
                  PackDetInputs(meta_keys=(
                        'img_id',
                        'img_path',
                        'ori_shape',
                        'img_shape',
                        'flip',
                        'flip_direction',
                    ))
                 ]


#Notice: due to jittor has limitted support for hooks,so the hook used will be added into train logic
#todo


# In[ ]:





# In[51]:


train_pipeline = ([
    LoadImageFromFile(backend_args=None),
    LoadYOLOAnnotations(with_bbox=True),
    YOLO_Mosaic(img_scale=(
                    640,
                    640,
                ),
                pad_val=114.0,
                pre_transform=[
                    LoadImageFromFile(backend_args=None),
                    LoadYOLOAnnotations(with_bbox=True),
                ]),
    YOLOv5RandomAffine(border=(
                    -320,
                    -320,
                ),
                border_val=(
                    114,
                    114,
                    114,
                ),
                max_aspect_ratio=100,
                max_rotate_degree=0.0,
                max_shear_degree=0.0,
                scaling_ratio_range=(
                    0.5,
                    1.5,
                )),
    Albu(bbox_params=dict(
                    format='pascal_voc',
                    label_fields=[
                        'gt_bboxes_labels',
                        'gt_ignore_flags',
                    ],
                    type='BboxParams'),
                keymap=dict(gt_bboxes='bboxes', img='image'),
                transforms=[
                    dict(p=0.01, type='Blur'),
                    dict(p=0.01, type='MedianBlur'),
                    dict(p=0.01, type='ToGray'),
                    dict(p=0.01, type='CLAHE'),
                ]),
    YOLOv5HSVRandomAug,
    RandomFlip(prob=0.5),
    PackDetInputs(meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'flip',
                    'flip_direction',
                ))
    ])
test_pipeline = [
    LoadImageFromFile(backend_args = None),
    YOLOv5KeepRatioResize(scale=(
                640,
                640,
            )),
    LetterResize(allow_scale_up=False,
                pad_val=dict(img=114),
                scale=(
                    640,
                    640,
                )),
    LoadYOLOAnnotations(with_bbox=True),
    PackDetInputs(meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                    'pad_param',
                ))
]


# In[ ]:





# In[52]:


class module(Module):
    pass


# In[ ]:





# In[ ]:





# In[53]:


train_dataset = YOLOv5CocoDataset(
    ann_file = train_ann_file,
    data_prefix = dict(img=train_data_prefix),
    data_root = data_root,
    filter_cfg=dict(filter_empty_gt=False, min_size=32),
    metainfo=dict(classes=(
            'car',
            'truck',
            'bus',
        )),
    pipeline = train_pipeline
)
val_dataset = YOLOv5CocoDataset(
    test_mode=True,
    pipeline = test_pipeline,
    ann_file=val_ann_file,
        batch_shapes_cfg=None,
        data_prefix=dict(img=val_data_prefix),
        data_root=data_root,
        metainfo=dict(classes=(
            'car',
            'truck',
            'bus',
        )),
)


# In[ ]:





# In[54]:


if __name__ == '__main__':
    mp.set_start_method('fork',force = True)

