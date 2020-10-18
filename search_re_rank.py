################################################
# grid search the best parameters of re-ranking
# Author: Jiaxuan Lu
################################################

from __future__ import print_function
from __future__ import division

import os
import os.path as osp
import time
import argparse
import numpy as np
import json

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from torchreid import data_manager, metrics
from torchreid.dataset_loader import ImageDataset
from torchreid import transforms as T
from torchreid import models
from torchreid.utils.avgmeter import AverageMeter
from torchreid.utils.torchtools import count_num_param
# from torchreid.utils.re_ranking import re_ranking
from torchreid.utils.reranking import re_ranking
from torchreid.eval_metrics import evaluate
from torchreid.utils.reidtools import visualize_ranked_results


re_rank_params = [(20,4,0.3),
                  (20,5,0.3),
                  (20,6,0.3),
                  (30,4,0.8)
                  ]

gpu_devices = "7" # gpu devices
model_weight = "./log/resnet50-xent/vmgn_hgnn14/checkpoint_ep80.pth.tar"
vis_ranked_res = True # save the visual results or not
save_dir = "./img/vmgn_hgnn14_ranked_results"
arch = "vmgn_hgnn"
re_rank = True
test_batch = 500
dataset_name = "pclreid"
global_branch = True
dist_metric = "cosine"
root = "./"
height = 256
width = 128
seed = 1
workers = 0

def main():
    torch.manual_seed(seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_devices
    use_gpu = torch.cuda.is_available()

    if use_gpu:
        print("Currently using GPU {}".format(gpu_devices))
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(seed)
    else:
        print("Currently using CPU (GPU is highly recommended)")

    print("Initializing dataset {}".format(dataset_name))
    dataset = data_manager.init_imgreid_dataset(
        root=root, name=dataset_name, split_id=0, isFinal=False,
        cuhk03_labeled=False, cuhk03_classic_split=False,
    )

    transform_test = T.Compose([
        T.Resize((height, width)),
        T.ToTensor(),
        # T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        T.Normalize(mean=[0.3495, 0.3453, 0.3941], std=[0.2755, 0.2122, 0.2563]),
    ])

    pin_memory = True if use_gpu else False

    queryloader = DataLoader(
        ImageDataset(dataset.query, transform=transform_test, isFinal=False),
        batch_size=test_batch, shuffle=False, num_workers=workers,
        pin_memory=pin_memory, drop_last=False
    )

    galleryloader = DataLoader(
        ImageDataset(dataset.gallery, transform=transform_test, isFinal=False),
        batch_size=test_batch, shuffle=False, num_workers=workers,
        pin_memory=pin_memory, drop_last=False
    )

    print("Initializing model: {}".format(arch))

    '''
           vmgn_hgnn model, arch chosen from {'resnet50','resnet101','resnet152'}
           efficientnet_hgnn model, arch chosen from {'efficientnet-b0', 'efficientnet-b1', 'efficientnet-b2', 'efficientnet-b3',
           'efficientnet-b4', 'efficientnet-b5', 'efficientnet-b6', 'efficientnet-b7','efficientnet-b8'}
           '''
    model = models.init_model(name=arch,
                              # num_classes=29626, # 29626 or 34394
                              num_classes=19658,
                              isFinal=False,
                              global_branch=global_branch,
                              arch="resnet50")
    print("Model size: {:.3f} M".format(count_num_param(model)))

    checkpoint = torch.load(model_weight)
    pretrain_dict = checkpoint['state_dict']
    # model_dict = model.state_dict()
    # pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict and model_dict[k].size() == v.size()}
    # model_dict.update(pretrain_dict)
    model.load_state_dict(pretrain_dict)

    if use_gpu:
        model = nn.DataParallel(model).cuda()

    print("Evaluate only")
    distmat = test(model, queryloader, galleryloader, use_gpu, return_distmat=True)
    if vis_ranked_res:
        visualize_ranked_results(
            distmat, dataset,
            save_dir=osp.join(save_dir),
            topk=20,
        )



def test(model, queryloader, galleryloader, use_gpu, ranks=[1, 5, 10, 20], return_distmat=False):
    batch_time = AverageMeter()

    model.eval()

    with torch.no_grad():
        qf, q_pids, q_camids = [], [], []
        for batch_idx, (imgs, pids, camids) in enumerate(queryloader):
            if use_gpu:
                imgs = imgs.cuda()

            end = time.time()
            features = model(imgs)
            batch_time.update(time.time() - end)

            features = features.data.cpu()
            qf.append(features)
            q_pids.extend(pids)
            q_camids.extend(camids)
        qf = torch.cat(qf, 0)
        q_pids = np.asarray(q_pids)
        q_camids = np.asarray(q_camids)

        print("Extracted features for query set, obtained {}-by-{} matrix".format(qf.size(0), qf.size(1)))

        gf, g_pids, g_camids = [], [], []
        for batch_idx, (imgs, pids, camids) in enumerate(galleryloader):
            if use_gpu:
                imgs = imgs.cuda()

            end = time.time()
            features = model(imgs)
            batch_time.update(time.time() - end)

            features = features.data.cpu()
            gf.append(features)
            g_pids.extend(pids)
            g_camids.extend(camids)
        gf = torch.cat(gf, 0)
        g_pids = np.asarray(g_pids)
        g_camids = np.asarray(g_camids)

        print("Extracted features for gallery set, obtained {}-by-{} matrix".format(gf.size(0), gf.size(1)))

    print("==> BatchTime(s)/BatchSize(img): {:.3f}/{}".format(batch_time.avg, test_batch))

    # m, n = qf.size(0), gf.size(0)
    # distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
    #           torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    # distmat.addmm_(1, -2, qf, gf.t())

    # slow re-ranking
    # distmat = metrics.compute_distance_matrix(qf, gf, dist_metric)
    # distmat = distmat.numpy()

    for (k1, k2, lambda_value) in re_rank_params:
        if re_rank:
            print('Applying person re-ranking with k1 = {}, k2 = {}, labmda = {}'.format(k1, k2, lambda_value))
            # slow re-ranking
            # distmat_qq = metrics.compute_distance_matrix(qf, qf, dist_metric)
            # distmat_gg = metrics.compute_distance_matrix(gf, gf, dist_metric)
            # distmat = re_ranking(distmat, distmat_qq, distmat_gg, k1=k1, k2=k2, lambda_value=lambda_value)
            # fast re-ranking
            distmat = re_ranking(qf, gf, k1=k1, k2=k2, lambda_value=lambda_value)

        print("Computing CMC and mAP")
        cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids, use_metric_cuhk03=False)

        print("Results ----------")
        print("mAP: {:.1%}".format(mAP))
        print("CMC curve")
        for r in ranks:
            print("Rank-{:<3}: {:.1%}".format(r, cmc[r - 1]))
        print("------------------")

    if return_distmat:
        return distmat
    return cmc[ranks[0] - 1], cmc[ranks[1] - 1], cmc[ranks[2] - 1], cmc[ranks[3] - 1], mAP


main()