from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import glob
import re
import sys
import urllib
import tarfile
import zipfile
import os.path as osp
from scipy.io import loadmat
import numpy as np
import h5py
#from scipy.misc import imsave


class Pclreid(object):
    dataset_dir = 'PCL_ReID'

    def __init__(self, root='data', verbose=True,isFinal=False, extended_data=False,**kwargs):
        super(Pclreid, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.list_train_path = osp.join(self.dataset_dir, 'train_extended_list.txt') if extended_data else \
            osp.join(self.dataset_dir, 'train_list.txt')
        self.list_query_path = osp.join(self.dataset_dir, 'query_list_final.txt') if isFinal else \
            osp.join(self.dataset_dir, 'query_list.txt')
        self.list_gallery_path = osp.join(self.dataset_dir, 'gallery_list_final.txt') if isFinal else \
            osp.join(self.dataset_dir, 'gallery_list.txt')
        self._check_before_run()

        if isFinal:
            query, num_query_imgs = self._process_dir_final(self.list_query_path, cam=1)
            gallery, num_gallery_imgs = self._process_dir_final(self.list_gallery_path, cam=2)
        else:
            train, num_train_pids, num_train_imgs = self._process_dir(self.list_train_path,cam=0)
            query, num_query_pids, num_query_imgs = self._process_dir(self.list_query_path,cam=1)
            gallery, num_gallery_pids, num_gallery_imgs = self._process_dir(self.list_gallery_path,cam=2)
    
            num_total_pids = num_train_pids + num_query_pids
            num_total_imgs = num_train_imgs + num_query_imgs + num_gallery_imgs
    
            if verbose:
                print("=> PCL loaded")
                print("Dataset statistics:")
                print("  ------------------------------")
                print("  subset   | # ids | # images")
                print("  ------------------------------")
                print("  train    | {:5d} | {:8d}".format(num_train_pids, num_train_imgs))
                print("  query    | {:5d} | {:8d}".format(num_query_pids, num_query_imgs))
                print("  gallery  | {:5d} | {:8d}".format(num_gallery_pids, num_gallery_imgs))
                print("  ------------------------------")
                print("  total    | {:5d} | {:8d}".format(num_total_pids, num_total_imgs))
                print("  ------------------------------")
                
            self.num_train_pids = num_train_pids
            self.num_query_pids = num_query_pids
            self.num_gallery_pids = num_gallery_pids
            self.train = train

        self.query = query
        self.gallery = gallery


    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))

    def _process_dir(self, list_path,cam):
        with open(list_path, 'r') as txt:
            lines = txt.readlines()
        dataset = []
        pid_container = set()
        for img_idx, img_info in enumerate(lines):

            img_path, pid = img_info.split(':')
            pid = int(pid) # no need to relabel

            camid = cam
            # img_path = osp.join(dir_path, img_path)
            dataset.append((img_path, pid, camid))
            pid_container.add(pid)
        num_imgs = len(dataset)
        num_pids = len(pid_container)
        # check if pid starts from 0 and increments with 1

        return dataset, num_pids, num_imgs

    def _process_dir_final(self, list_path,cam):
        with open(list_path, 'r') as txt:
            lines = txt.readlines()
        dataset = []
        for img_idx, img_info in enumerate(lines):

            img_path = img_info.replace("\n","")
            camid = cam
            # img_path = osp.join(dir_path, img_path)
            dataset.append((img_path, camid))
        num_imgs = len(dataset)

        return dataset, num_imgs