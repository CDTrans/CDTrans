#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/1/17 15:00
# @Author  : Hao Luo
# @File    : msmt17.py

import glob
import re

import os.path as osp

from .bases import BaseImageDataset


class OfficeHome(BaseImageDataset):
    """
    Office Home
    """
    dataset_dir = ''

    def __init__(self, root_train='./datasets/reid_datasets/Corrected_Market1501', root_val='./datasets/reid_datasets/Corrected_Market1501', pid_begin=0, verbose=True, **kwargs):
        super(OfficeHome, self).__init__()
        root_train = root_train
        root_valid = root_val
        self.train_dataset_dir = osp.dirname(root_train)
        self.valid_dataset_dir = osp.dirname(root_val)
        self.train_name = osp.basename(root_train).split('.')[0]
        self.valid_name = osp.basename(root_valid).split('.')[0]
        self.pid_begin = pid_begin
        train = self._process_dir(root_train, self.train_dataset_dir)
        valid = self._process_dir(root_valid, self.valid_dataset_dir)

        
        if verbose:
            print("=> Office-Home loaded")
            self.print_dataset_statistics(train, valid)
            
        self.train = train
        self.valid = valid
        self.test = valid

        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = self.get_imagedata_info(self.train)   
        self.num_valid_pids, self.num_valid_imgs, self.num_valid_cams, self.num_valid_vids = self.get_imagedata_info(self.valid)
        self.num_test_pids = self.num_valid_pids

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.art_dir):
            raise RuntimeError("'{}' is not available".format(self.art_dir))
        if not osp.exists(self.clipart_dir):
            raise RuntimeError("'{}' is not available".format(self.clipart_dir))
        if not osp.exists(self.product_dir):
            raise RuntimeError("'{}' is not available".format(self.product_dir))
        if not osp.exists(self.realworld_dir):
            raise RuntimeError("'{}' is not available".format(self.realworld_dir))

    def print_dataset_statistics(self, train, valid):
        num_train_pids, num_train_imgs, num_train_cams, num_train_views = self.get_imagedata_info(train)
        num_valid_pids, num_valid_imgs, num_valid_cams, num_targe_views = self.get_imagedata_info(valid)

        print("Dataset statistics:")
        print("train {} and valid is {}".format(self.train_name, self.valid_name))
        print("  ----------------------------------------")
        print("  subset   | # ids | # images | # cameras")
        print("  ----------------------------------------")
        print("  train   | {:5d} | {:8d} | {:9d}".format( num_train_pids, num_train_imgs, num_train_cams))
        print("  valid   | {:5d} | {:8d} | {:9d}".format(num_valid_pids, num_valid_imgs, num_valid_cams))
        print("  ----------------------------------------")
        
    def _process_dir(self, list_path, dir_path):
        with open(list_path, 'r') as txt:
            lines = txt.readlines()
        dataset = []
        pid_container = set()
        cam_container = set()
        for img_idx, img_info in enumerate(lines):
            img_path, pid = img_info.split(' ')
            pid = int(pid)  # no need to relabel
            img_path = osp.join(dir_path, img_path)
            dataset.append((img_path, self.pid_begin +pid, 0, 0, img_idx))
            pid_container.add(pid)
#             cam_container.add(camid)
#         print(cam_container, 'cam_container')
        # check if pid starts from 0 and increments with 1
        for idx, pid in enumerate(pid_container):
            assert idx == pid, "See code comment for explanation"
        return dataset